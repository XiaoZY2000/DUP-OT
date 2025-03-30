import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReviewDataset(Dataset):
    def __init__(self, reviews_dict):
        """
        :param reviews_dict: {node_id: {"word_embeddings": [word_embedding_1, word_embedding_2, ...], "sentence_embedding": , "ratings": [1.0, 2.0, 3.0, ...]}}
        """
        self.embeddings = [] # 每个单词的embedding
        self.sentence_embeddings = [] # 整体的embedding
        self.ratings = [] # 评分序列

        for info_dict in reviews_dict.values():
            embed = info_dict["word_embeddings"]
            sentence_emb = info_dict["sentence_embedding"]
            rating = info_dict["ratings"]
            embed = torch.tensor(embed, dtype=torch.float32).to(device) # (num_of_words, emb_dim)
            sentence_emb = torch.tensor(sentence_emb, dtype=torch.float32).to(device) # (emb_dim)
            rating = torch.tensor(rating, dtype=torch.float32).to(device) # (num_of_items)
            self.embeddings.append(embed)
            self.sentence_embeddings.append(sentence_emb)
            self.ratings.append(rating)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.sentence_embeddings[index], self.ratings[index]

def collate_fn(batch, max_length=500, max_items=20):
    """
    处理变长 `embedding` 矩阵，确保输出固定长度
    :param batch: List of (num_words, embedding_dim) 变长的 `embedding`
    :param max_length: 设定的最大长度
    :return: (batch_size, max_length, embedding_dim), (batch_size, max_length)
    """
    embedding_dim = batch[0].shape[1]  # 获取 embedding 维度
    processed_batch = []
    processed_rating = []
    _, sentence_embeddings, _ = zip(*batch)

    for emb, _, rating in batch:
        num_words = emb.shape[0]  # 该评论的单词数
        num_items = rating.shape[0]

        # **截断**
        if num_words > max_length:
            emb = emb[:max_length, :]

        # **填充**
        elif num_words < max_length:
            pad_size = (0, 0, 0, max_length - num_words)  # (left_pad, right_pad, top_pad, bottom_pad)
            emb = F.pad(emb, pad_size, "constant", 0)  # (max_length, embedding_dim)

        if num_items > max_items:
            rating = rating[:max_items]

        elif num_items < max_items:
            pad_size = (0, max_items - num_items)
            rating = F.pad(rating, pad_size, "constant", 0)

        processed_batch.append(emb)
        processed_rating.append(rating)

    # **拼接为 batch 张量**
    reviews_padded = torch.stack(processed_batch)  # (batch_size, max_length, embedding_dim)
    reviews_mask = (reviews_padded.abs().sum(dim=-1) > 1e-6).float()  # 生成 Mask（填充部分为 0）
    ratings_padded = torch.stack(processed_rating) # (batch_size, max_items)
    ratings_mask = (ratings_padded != 0.0).float()
    sentence_embeddings = torch.stack(sentence_embeddings)

    return reviews_padded, reviews_mask, ratings_padded, ratings_mask, sentence_embeddings

class MinMaxNorm(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super(MinMaxNorm, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x_min = x.min(dim=-1, keepdim=True)[0]  # 计算每个样本的最小值
        x_max = x.max(dim=-1, keepdim=True)[0]  # 计算每个样本的最大值
        return (x - x_min) / (x_max - x_min + 1e-8) * (self.max_val - self.min_val) + self.min_val

class RatingEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=100):
        super(RatingEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, ratings_padded, ratings_mask):
        # 计算评分 embedding
        embedding = self.mlp(ratings_padded.unsqueeze(-1))  # (batch_size, num_scores, output_dim)

        # **计算有效评分的加权平均**
        weighted_embedding = (embedding * ratings_mask.unsqueeze(-1)).sum(dim=1) / (ratings_mask.sum(dim=1, keepdim=True) + 1e-8)

        return weighted_embedding  # (batch_size, output_dim)

class RatingDecoder(nn.Module):
    def __init__(self, aspect_dim=100, max_items=20, hidden_dim=50):
        super(RatingDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(aspect_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_items),
            MinMaxNorm(0, 5)
        )

    def forward(self, ratings_emb):
        ratings_recon = self.MLP(ratings_emb)
        return ratings_recon

class ReviewsEncoder(nn.Module):
    def __init__(self, emb_dim=768, target_dim=100, hidden_dim=200):
        super(ReviewsEncoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, sentence_emb):
        review_encode = self.MLP(sentence_emb)
        return review_encode

class ReviewsDecoder(nn.Module):
    def __init__(self, target_dim=100, emb_dim=768, hidden_dim=200):
        super(ReviewsDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, review_encode):
        review_recon = self.MLP(review_encode)
        return review_recon

class TextConv(nn.Module):
    def __init__(self, embedding_dim=768, num_filters=768, window_size=3):
        super(TextConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=num_filters,
                              kernel_size=window_size,
                              stride=1,
                              padding=1
        )

    def forward(self, reviews_padded):
        batch_size, num_words, emb_dim = reviews_padded.shape
        reviews_padded = reviews_padded.permute(0, 2, 1)
        reviews_padded = self.conv(reviews_padded)
        reviews_padded = reviews_padded.permute(0, 2, 1)

        return reviews_padded

class AspectGateControl(nn.Module):
    def __init__(self, embedding_dim=768, embedding_aspect=100, num_aspects=10):
        super(AspectGateControl, self).__init__()
        self.Wm = nn.Parameter(torch.randn(num_aspects, embedding_aspect, embedding_dim))  # (num_aspects, emb_aspect, emb_dim)
        self.bm = nn.Parameter(torch.randn(num_aspects, 1, embedding_aspect))  # (num_aspects, 1, emb_aspect)
        self.Wmg = nn.Parameter(torch.randn(num_aspects, embedding_aspect, embedding_dim))  # (num_aspects, emb_aspect, emb_dim)
        self.bmg = nn.Parameter(torch.randn(num_aspects, 1, embedding_aspect))  # (num_aspects, 1, emb_aspect)
        self.sigmoid = nn.Sigmoid()

    def forward(self, reviews_conv):
        """
        :param reviews_conv: (batch_size, num_words, emb_dim)
        :param reviews_mask: (batch_size, num_words)  # mask for valid words (optional)
        :return: output (batch_size, num_aspects, num_words, embedding_aspect)
        """
        batch_size, num_words, emb_dim = reviews_conv.shape

        # **计算 term_1**
        term_1 = torch.einsum('ade, bwe->bawd', self.Wm, reviews_conv) + self.bm  # (batch_size, num_aspects, num_words, embedding_aspect)

        # **计算 term_2（门控机制）**
        term_2 = self.sigmoid(torch.einsum('ade, bwe->bawd', self.Wmg, reviews_conv) + self.bmg)

        # **最终输出**
        output = term_1 * term_2  # (batch_size, num_aspects, num_words, embedding_aspect)

        return output

class AspectAttention(nn.Module):
    def __init__(self, num_aspects=10, embedding_aspect=100):
        super(AspectAttention, self).__init__()
        self.V = nn.Parameter(torch.randn(num_aspects, embedding_aspect))  # (num_aspects, embedding_aspect)

    def forward(self, g_mju, reviews_mask):
        """
        :param g_mju: (batch_size, num_aspects, num_words, embedding_aspect)
        :param reviews_mask: (batch_size, num_words)  # 1 表示有效单词，0 表示 padding
        :return: a_m_u (batch_size, num_aspects, embedding_aspect)
        """
        # **计算 `energy = g_{m, j, u}^{\top} V_{m, s}`**
        energy = torch.einsum('bawd,ad->baw', g_mju, self.V)  # (batch_size, num_aspects, num_words)

        # **扩展 `reviews_mask` 到 `energy` 形状**
        mask = reviews_mask.unsqueeze(1).expand_as(energy)  # (batch_size, num_aspects, num_words)

        # **将 `mask == 0` 的地方设置为 `-inf`，让 softmax 忽略这些位置**
        energy = energy.masked_fill(mask == 0, float('-inf'))

        # **计算 `β_{m, j, u}`**
        beta = torch.softmax(energy, dim=-1)  # (batch_size, num_aspects, num_words)

        # **计算 `a_{m, u} = Σ β_{m, j, u} g_{m, j, u}`**
        a_m_u = torch.einsum('baw, bawd->bad', beta, g_mju)  # (batch_size, num_aspects, embedding_aspect)

        return a_m_u

class UserAspectEmbeddingDecoder(nn.Module):
    def __init__(self, num_aspects=10, embedding_aspect=100, hidden_dim=500):
        super(UserAspectEmbeddingDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_aspects*embedding_aspect, hidden_dim=500),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_aspect)
        )

    def forward(self, a_m_u):
        # a_m_u: (batch_size, num_aspects, embedding_aspect)
        batch_size = a_m_u.shape[0]
        a_m_u = a_m_u.view(batch_size, -1)
        reconstructed_node = self.decoder(a_m_u)
        return reconstructed_node

class RatingReconstructDecoder(nn.Module):
    def __init__(self, num_aspects=10, embedding_aspect=100, max_items=20, hidden_dim=500, embedding_dim=768):
        super(RatingReconstructDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_aspects*embedding_aspect, hidden_dim=500),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_items),
            MinMaxNorm(0, 5)
        )

    def forward(self, a_m_u):
        batch_size = a_m_u.shape[0]
        a_m_u = a_m_u.view(batch_size, -1)
        reconstructed_rating = self.decoder(a_m_u)
        return reconstructed_rating

class AspectExtractionModel(nn.Module):
    def __init__(self, num_aspects=10, max_items=20, embedding_dim=768, embedding_aspect=100, hidden_dim=500):
        super(AspectExtractionModel, self).__init__()
        self.textconv = TextConv()
        self.aspectgate = AspectGateControl(embedding_dim=embedding_dim, embedding_aspect=embedding_aspect, num_aspects=num_aspects)
        self.aspectattention = AspectAttention(num_aspects=num_aspects, embedding_aspect=embedding_aspect)
        self.useraspectdecoder = UserAspectEmbeddingDecoder(num_aspects=num_aspects, embedding_aspect=embedding_aspect, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        self.ratingreconstructdecoder = RatingReconstructDecoder(num_aspects, embedding_aspect, max_items, hidden_dim, embedding_dim)

    def forward(self, reviews_padded, reviews_mask):
        x = self.textconv(reviews_padded)
        x = self.aspectgate(x)
        aspect_level_embedding = self.aspectattention(x, reviews_mask)
        reconstructed_embedding = self.useraspectdecoder(aspect_level_embedding)
        reconstructed_rating = self.ratingreconstructdecoder(aspect_level_embedding)
        return aspect_level_embedding, reconstructed_embedding, reconstructed_rating

class OverallModel(nn.Module):
    def __init__(self, num_aspects=10, max_items=20, embedding_dim=768, aspect_dim=100, hidden_dim=500):
        super(OverallModel, self).__init__()
        self.ratingencoder = RatingEncoder(input_dim=1, hidden_dim=50, output_dim=aspect_dim)
        self.ratingdecoder = RatingDecoder(aspect_dim=aspect_dim, max_items=max_items, hidden_dim=50)
        self.reviewencoder = ReviewsEncoder(emb_dim=embedding_dim, target_dim=aspect_dim, hidden_dim=200)
        self.reviewdecoder = ReviewsDecoder(aspect_dim=aspect_dim, max_items=max_items, hidden_dim=50)
        self.aspectextractor = AspectExtractionModel(num_aspects=num_aspects, max_items=max_items, embedding_dim=embedding_dim, embedding_aspect=aspect_dim, hidden_dim=hidden_dim)

    def forward(self, reviews_padded, reviews_mask, ratings_padded, ratings_mask, sentence_emb):
        inter_rating_emb = self.ratingencoder(ratings_padded, ratings_mask)
        inter_sentence_emb = self.reviewencoder(sentence_emb)
        recon_rating = self.ratingdecoder(inter_rating_emb)
        recon_sentence = self.reviewdecoder(inter_sentence_emb)
        aspect_level_embedding, reconstructed_embedding, reconstructed_rating = self.aspectextractor(reviews_padded, reviews_mask)

        return inter_rating_emb, inter_sentence_emb, recon_rating, recon_sentence, aspect_level_embedding, reconstructed_embedding, reconstructed_rating
        
class ReconstructLoss(nn.Module):
    def __init__(self):
        super(ReconstructLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_loss_keep = nn.MSELoss(reduction='none')  # 用于 mask 计算

    def forward(self, recon_sentence, inter_sentence_emb, sentence_emb, 
                recon_rating, inter_rating_emb, ratings_padded, ratings_mask, 
                reconstructed_embedding, reconstructed_rating):

        # **Loss 1: 句子级别 MSE**
        loss_1 = self.mse_loss(recon_sentence, sentence_emb)  

        # **Loss 2: 评分级别 MSE + Mask**
        loss_2 = self.mse_loss_keep(recon_rating, ratings_padded)  # (batch_size, num_items)
        masked_loss_2 = loss_2 * ratings_mask  # 只计算有效评分部分
        loss_2 = masked_loss_2.sum() / (ratings_mask.sum() + 1e-8)  # 防止除零

        # **Loss 3: 用户意图 embedding 还原**
        loss_3 = self.mse_loss(inter_sentence_emb, reconstructed_embedding)

        # **Loss 4: 用户评分 embedding 还原**
        loss_4 = self.mse_loss(inter_rating_emb, reconstructed_rating)

        # **最终损失**
        loss = loss_1 + loss_2 + loss_3 + loss_4
        return loss


def train_model(dataset, model, loss_fn, optimizer, epochs=2000, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    total_loss_list = []
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for reviews_padded, reviews_mask, ratings_padded, ratings_mask, sentence_embeddings in tqdm(dataloader, desc='Training on Epoch: ' + str(epoch)):
            optimizer.zero_grad()
            inter_rating_emb, inter_sentence_emb, recon_rating, recon_sentence, aspect_level_embedding, reconstructed_embedding, reconstructed_rating = model(reviews_padded, reviews_mask, ratings_padded, ratings_mask, sentence_embeddings)
            loss = loss_fn(recon_sentence, inter_sentence_emb, sentence_embeddings, recon_rating, inter_rating_emb, ratings_padded, ratings_mask, reconstructed_embedding, reconstructed_rating)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= len(dataloader)
        total_loss_list.append(total_loss)
        if epoch%10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")
            
    plt.figure(figsize=(8,6))
    plt.plot(list(range(len(total_loss_list))), total_loss_list, label='Train Loss', color='b', linestyle='-')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("Training_loss_curve.png")

    return model

def train(reviews_dict):
    dataset = ReviewDataset(reviews_dict)
    model = OverallModel(num_aspects=10, max_items=20, embedding_dim=768, aspect_dim=100, hidden_dim=500).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    loss_fn = ReconstructLoss()
    model_path = './model_save/overall_model.pth'

    if not os.path.exists(model_path):
        model = train_model(dataset, model, loss_fn, optimizer, epochs=2000, batch_size=64)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))