import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os
import joblib
import matplotlib.pyplot as plt

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

overwrite = False

class UserInteractionsDataset(Dataset):
    def __init__(self, user_infos):
        # 用户信息：{user_1: {"embedding": user_embedding, "interactions": interacted_item_embeddings, "ratings": ratings}}
        self.user_infos = user_infos

    def __len__(self):
        return len(list(self.user_infos.keys()))

    def __getitem__(self, index):
        user = list(self.user_infos.keys())[index]
        user_embedding = torch.tensor(self.user_infos[user]["embedding"]).to(device).to(dtype)
        item_embeddings = torch.tensor(self.user_infos[user]["interactions"]).to(device).to(dtype)
        ratings = torch.tensor(self.user_infos[user]["ratings"]).to(device).to(dtype)
        
        return user_embedding, item_embeddings, ratings

def collate_fn(batch, max_items=20):
    user_embedding_list, _, _ = zip(*batch)

    processed_items = []
    processed_ratings = []

    for user_embedding, item_embeddings, ratings in batch:
        num_items = ratings.shape[0]
        if num_items > max_items:
            ratings = ratings[:max_items]
            item_embeddings = item_embeddings[:max_items]

        elif num_items < max_items:
            pad_size_items = (0, 0, 0, max_items - num_items)
            pad_size_ratings = (0, max_items - num_items)
            item_embeddings = F.pad(item_embeddings, pad_size_items, "constant", 0)
            ratings = F.pad(ratings, pad_size_ratings, "constant", 0)

        processed_items.append(item_embeddings)
        processed_ratings.append(ratings)

    items_padded = torch.stack(processed_items) # (batch_size, max_items, emb_dim)
    items_mask = (items_padded.abs().sum(dim=-1) > 1e-6).float() # (batch_size, max_items)
    ratings_padded = torch.stack(processed_ratings) # (batch_size, max_items)
    user_embedding = torch.stack(user_embedding_list)

    return user_embedding, items_padded, items_mask, ratings_padded

# 用用户的特征输入MLP计算权重，MLP参数可调
class UserGMMModel(nn.Module):
    def __init__(self, feature_dim, num_components):
        super(UserGMMModel, self).__init__()
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.hidden_dim = feature_dim // 2
        
        # 用户对 GMM 组件的权重 (K 维向量)
        self.user_weights = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_components),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, user_embeddings):
        batch_size = user_embeddings.shape[0]
        
        # 用户个性化 GMM 权重
        w_u = self.user_weights(user_embeddings)  # (batch_size, K)

        return w_u

class RatingPredictor(nn.Module):
    def __init__(self, K):
        super(RatingPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(K, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, item_densities):
        batch_size, K, max_items = item_densities.shape
        x = item_densities.permute(0, 2, 1)
        x = x.reshape(-1, K) # (batch_size*max_items, K)
        rating = self.mlp(x) # (batch_size*max_items, 1)
        rating = rating.view(batch_size, max_items)
        return rating

def mahalanobis_distance_full(items_padded, items_mask, mu, sigma_inv):
    # diff: (B, M, D)
    diff = items_padded - mu.view(1, 1, -1)

    # (x - mu)^T * Sigma^{-1} * (x - mu):  → (B, M)
    mahalanobis_sq = torch.einsum("bni,ij,bnj->bn", diff, sigma_inv, diff)

    return mahalanobis_sq * items_mask

def gaussian_density(items_padded, items_mask, mu, sigma, sigma_inv):
    # 以batch为单位计算物品在某一高斯分布上的概率密度，输入为(batch_size, max_items, feature_dim)，输出为(batch_size, max_items)
    feature_dim = items_padded.shape[-1]

    # 计算 (x - mu)
    diff = items_padded - mu.view(1, 1, -1)  # (batch_size, max_items, feature_dim)

    # 计算 (x - mu)^T * Sigma^{-1} * (x - mu)
    mahalanobis_term = torch.einsum("bni,ij,bnj->bn", diff, sigma_inv, diff) # (batch_size, max_items)
    exp_term = -0.5 * mahalanobis_term  # (batch_size, max_items)

    # 计算归一化项 (norm_term)
    det_sigma = torch.det(sigma)
    norm_term = torch.sqrt((2 * np.pi) ** feature_dim * det_sigma).unsqueeze(0) # (1, 1)

    # 计算最终概率密度
    return (torch.exp(exp_term) / norm_term) * items_mask  # (batch_size, max_items)

def predict_rating(user_embeddings, items_padded, items_mask, mu_k, sigma_inv_k, model, rating_predictor):
    """计算用户对物品的评分 (使用概率密度计算)"""
    batch_size = user_embeddings.shape[0]
    w_u = model(user_embeddings)  # 获取用户的 GMM 权重 (batch_size, K)
    # print("user weights (mean/std):", w_u.mean().item(), w_u.std().item())
    K = w_u.shape[-1]

    p_overall = []
    
    for k in range(K):
        # 计算物品在每一个高斯分布上的概率密度
        p_k = mahalanobis_distance_full(items_padded, items_mask, mu_k[k], sigma_inv_k[k]) # (batch_size, max_items)
        p_k = torch.clamp(p_k, min=0.0, max=100.0) / 100.0  # 简单线性缩放
        p_overall.append(p_k)

    p_d_K = torch.stack(p_overall, dim=1) # (batch_size, K, max_items)
    p_weighted = w_u.unsqueeze(-1) * p_d_K
    predicted_ratings = rating_predictor(p_weighted) # (batch_size, max_items)

    return predicted_ratings

def predict_rating_with_weights(w_u, item_embeddings, mu_k, sigma_inv_k, rating_predictor):
    K = w_u.shape[-1]

    p_overall = []

    mask = (item_embeddings.abs().sum(dim=-1) > 1e-6).float()
    for k in range(K):
        p_k = mahalanobis_distance_full(item_embeddings, mask, mu_k[k], sigma_inv_k[k])
        p_k = torch.clamp(p_k, min=0.0, max=100.0) / 100.0  # 简单线性缩放
        p_overall.append(p_k)

    p_d_K = torch.stack(p_overall, dim=1) # (1, K, num_items)

    p_weighted = w_u.unsqueeze(-1) * p_d_K
    predicted_ratings = rating_predictor(p_weighted) # (1, num_items)

    return predicted_ratings # (1, num_items)

def train_model(dataset, model, rating_predictor, optimizer, mu_k, sigma_k, sigma_inv_k, domain_str, epochs=2000, batch_size=128):
    """
    interaction_mask: (N, M) 的二值矩阵，表示哪些用户对哪些物品有评分
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    total_loss_list = []
    loss_fn = nn.MSELoss(reduction="none").to(device)  # 不要自动求和，便于 Mask 操作
    
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        rating_predictor.train()
        for user_embedding, items_padded, items_mask, ratings_padded in dataloader:
            optimizer.zero_grad()
            
            # 预测评分
            predicted_ratings = predict_rating(user_embedding, items_padded, items_mask, mu_k, sigma_inv_k, model, rating_predictor)
    
            # 仅计算有交互项的损失
            loss_matrix = loss_fn(predicted_ratings, ratings_padded)  # (batch_size, max_items)
            masked_loss = loss_matrix * items_mask  # 只计算交互部分
            loss = masked_loss.sum() / items_mask.sum()  # 归一化
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            # print("grad norm user_weights:", model.user_weights[0].weight.grad.norm().item())
            # print("grad rating_predictor weights:", rating_predictor.mlp[0].weight.grad.norm().item())
            # print("predicted_ratings stats:", predicted_ratings.min().item(), predicted_ratings.max().item())
    
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

        total_loss_list.append(total_loss/len(dataloader))
        
    plt.figure(figsize=(8,6))
    plt.plot(list(range(epochs)), total_loss_list, label='Train Loss', color='b', linestyle='-')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("User Weights Training Loss Curve")
    plt.legend()
    plt.savefig("userweightstraining_" + domain_str + ".png")

    return model, rating_predictor

def train(user_infos, item_features, task_name, domain_str, K):
    # 将K设为hyper parameter
    optimal_K = K
    # # 聚类动态计算K
    # print("Clustering")
    # K_range = range(100, 200)
    # silhouette_scores = []
    
    # for K in tqdm(K_range):
    #     kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    #     labels = kmeans.fit_predict(item_features)
    #     score = silhouette_score(item_features, labels)
    #     silhouette_scores.append(score)
    
    # # 选择轮廓系数最高的 K
    # optimal_K = K_range[np.argmax(silhouette_scores)]
    # print("Done")
    # print("聚类数量：", optimal_K)
    
    # 训练用户的个性化 GMM 参数
    model = UserGMMModel(feature_dim=item_features.shape[1], num_components=optimal_K).to(device)
    rating_predictor = RatingPredictor(optimal_K).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(rating_predictor.parameters()), lr=0.0001)

    model_path = './model_save/usergmm_model_' + task_name + "_" + domain_str + '.pth'
    rating_predictor_path = './model_save/ratingpredictor_' + task_name + "_" + domain_str + '.pth'
    mu_path = './model_save/mu_' + task_name + "_" + domain_str + '.pth'
    sigma_path = './model_save/sigma_' + task_name + "_" + domain_str + '.pth'
    sigma_inv_path = './model_save/sigma_inv_' + task_name + "_" + domain_str + '.pth'

    if not os.path.exists(model_path) or overwrite:
        # 训练 GMM
        gmm_path = './stored_files/gmm_' + task_name + '_' + domain_str + '.pkl'
        if not os.path.exists(gmm_path) or overwrite:
            print("Training GMM ...")
            gmm = GaussianMixture(n_components=optimal_K, covariance_type='diag', random_state=42)
            gmm.fit(item_features)
            joblib.dump(gmm, gmm_path)
            print("Done")
        else:
            gmm = joblib.load(gmm_path)
            print("GMM Loaded")
        
        # 获取 GMM 组件参数
        mu_k = torch.tensor(gmm.means_, dtype=torch.float32).to(device)  # (K, D)
        sigma_k = torch.tensor(gmm.covariances_, dtype=torch.float32).to(device)  # (K, D)
        # 从 (K, D) 构造 (K, D, D) 对角矩阵
        sigma_k = torch.stack([torch.diag(sigma_k[i]) for i in range(sigma_k.shape[0])])

        # 计算协方差矩阵的逆备用
        sigma_inv_k = torch.inverse(sigma_k) # (K, D, D)
    
        dataset = UserInteractionsDataset(user_infos)
        
        model, rating_predictor = train_model(dataset, model, rating_predictor, optimizer, mu_k, sigma_k, sigma_inv_k, domain_str, epochs=200, batch_size=128)
        torch.save(model.state_dict(), model_path)
        torch.save(rating_predictor.state_dict(), rating_predictor_path)
        torch.save(mu_k, mu_path)
        torch.save(sigma_k, sigma_path)
        torch.save(sigma_inv_k, sigma_inv_path)
    else:
        model.load_state_dict(torch.load(model_path))
        rating_predictor.load_state_dict(torch.load(rating_predictor_path))
        mu_k = torch.load(mu_path)
        sigma_k = torch.load(sigma_path)
        sigma_inv_k = torch.load(sigma_inv_path)

    return mu_k, sigma_k, sigma_inv_k, model, rating_predictor