import torch
import torch.nn.functional as F
import json
import numpy as np
import pickle
import os
from item_processing import embed, itememb_visualize, extract_user_and_item_reviews, split_random_rows, embed_all_items_fast 
from dataset_preprocess import cal_user_emb, build_userinfo, build_global_itemfeature
from distribution_calculation import train, predict_rating_with_weights
from optimal_transfer import wasserstein_gaussian, optimal_transport_matrix, cal_ot_matrix
from item_autoencoder import train_autoencoder
import nltk
from tqdm import tqdm

overwrite = False

# dataset_list = ['Movies_and_TV', 'Video_Games', 'Digital_Music', 'Movies_and_TV']
# dataset_list = ['Digital_Music', 'Video_Games', 'Kindle_Store', 'Movies_and_TV', 'Musical_Instruments']
dataset_list = ['Video_Games_5', 'Kindle_Store_5', 'CDs_and_Vinyl_5', 'Movies_and_TV_5']
# Gaussian_Distributions_num = {'Digital_Music': 32, 'Movies_and_TV': 32, 'Video_Games': 32, 'Kindle_Store': 510, 'Musical_Instruments': 202}
Gaussian_Distributions_num = {'Video_Games_5': 32, 'CDs_and_Vinyl_5': 32, 'Kindle_Store_5': 32, 'Movies_and_TV_5': 32}
# dataset_pairs = [['Digital_Music', 'Movies_and_TV'], ['Movies_and_TV', 'Digital_Music'], ['Digital_Music', 'Video_Games'], ['Video_Games', 'Digital_Music'], ['Movies_and_TV', 'Video_Games'], ['Video_Games', 'Movies_and_TV']]
# dataset_pairs = [['Video_Games', 'Digital_Music'], ['Kindle_Store','Video_Games'], ['Movies_and_TV','Video_Games'], ['Digital_Music', 'Musical_Instruments']]
dataset_pairs = [['CDs_and_Vinyl_5', 'Video_Games_5'], ['CDs_and_Vinyl_5', 'Kindle_Store_5'], ['CDs_and_Vinyl_5', 'Movies_and_TV_5'], ['Movies_and_TV_5', 'Video_Games_5'], ['Movies_and_TV_5', 'Kindle_Store_5']]

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding = 128

def main():
    user_item_review_dict = {}
    item_description_dict = {}
    user_item_mapping = {}
    user_item_rating = {}
    item_embedding_dict = {}
    redim_item_embedding_dict = {}
    user_embedding_dict = {}
    redim_user_embedding_dict = {}
    user_info = {}
    item_features = {}
    result_variable_dict = {}
    user_gmm_weights = {}
    user_reviews_dict = {}
    item_reviews_dict = {}
    task_names = {}
    nltk.download('punkt')
    nltk.download('punkt_tab')
    for dataset in dataset_list:
        # Read the dataset from the JSON file
        # data = {"interaction": {user1: {item1: {"review": , "rating": 5}, ...}, ...}, "metadata": {item1: {"description": "Title + Description", "images"}, ...}}
        filepath = './dataset/' + dataset + '.json'
        
        with open(filepath, 'r') as file:
            data = json.load(file)
        user_item_review_dict[dataset] = data
        user_item_mapping[dataset] = {user: list(user_item_review_dict[dataset][user].keys()) for user in user_item_review_dict[dataset].keys()}
        user_item_rating[dataset] = {user: [float(user_item_review_dict[dataset][user][item]["rating"]) for item in user_item_review_dict[dataset][user].keys()] for user in user_item_review_dict[dataset].keys()}

        user_reviews_dict[dataset], item_reviews_dict[dataset] = extract_user_and_item_reviews(user_item_review_dict[dataset])
        num_users = len(user_item_review_dict[dataset].keys())
        num_items = len(item_reviews_dict[dataset].keys())
        interactions = sum([len(user_item_mapping[dataset][user]) for user in user_item_mapping[dataset].keys()])
        density = interactions / (num_users * num_items)
        print("Num of items in " + dataset + ': ', num_items)
        print("Num of interactions " + dataset + ': ', interactions)
        print("Density of " + dataset + ': ', density)
        
        # 根据物品和用户评论获得物品和用户embedding
        # item_embedding_dict = {"Movies_and_TV": {item1: embedding1, item2: embedding2, ...}, ...}
        user_embedding_dict_path = "./stored_files/" + dataset + "_user_embedding_dict.pkl"
        item_embedding_dict_path = "./stored_files/" + dataset + "_item_embedding_dict.pkl"
        if not os.path.exists(item_embedding_dict_path) or overwrite:
            item_embedding_dict[dataset] = embed_all_items_fast(item_reviews_dict[dataset])
            with open(item_embedding_dict_path, 'wb') as f:
                pickle.dump(item_embedding_dict[dataset], f)
            print("Item Embedding Calculated")
        else:
            with open(item_embedding_dict_path, 'rb') as f:
                item_embedding_dict[dataset] = pickle.load(f)
            print("Item Embedding Loaded")

        if not os.path.exists(user_embedding_dict_path) or overwrite:
            user_embedding_dict[dataset] = embed_all_items_fast(user_reviews_dict[dataset])
            with open(user_embedding_dict_path, 'wb') as f:
                pickle.dump(user_embedding_dict[dataset], f)
            print("User Embedding Calculated")
        else:
            with open(user_embedding_dict_path, 'rb') as f:
                user_embedding_dict[dataset] = pickle.load(f)
            print("User Embedding Loaded")
            
    for dataset_pair in dataset_pairs:
        print("Preprocessing Stage for ", dataset_pair[0] + ' and ' + dataset_pair[1])
        task_name = dataset_pair[0] + "_" + dataset_pair[1]
        task_names[task_name] = dataset_pair
        redim_item_embedding_dict[task_name] = {}
        redim_user_embedding_dict[task_name] = {}
        shared_features = np.stack(list(item_embedding_dict[dataset_pair[0]].values()) + list(item_embedding_dict[dataset_pair[1]].values()) + list(user_embedding_dict[dataset_pair[0]].values()) + list(user_embedding_dict[dataset_pair[1]].values()), axis=0)
        source_item_features = np.stack(list(item_embedding_dict[dataset_pair[0]].values()), axis=0)
        target_item_features = np.stack(list(item_embedding_dict[dataset_pair[1]].values()), axis=0)
        source_user_features = np.stack(list(user_embedding_dict[dataset_pair[0]].values()), axis=0)
        target_user_features = np.stack(list(user_embedding_dict[dataset_pair[1]].values()), axis=0)
        source_item_features_sample = split_random_rows(source_item_features, 100)
        target_item_features_sample = split_random_rows(target_item_features, 100)
        # itememb_visualize(source_item_features_sample, target_item_features_sample)
        # breakpoint()
        # itememb_visualize(source_user_features, target_user_features)
        # breakpoint()
        model = train_autoencoder(shared_features, dataset_pair)
        model.eval()
        redim_item_embedding_dict[task_name][dataset_pair[0]] = {}
        redim_item_embedding_dict[task_name][dataset_pair[1]] = {}
        redim_user_embedding_dict[task_name][dataset_pair[0]] = {}
        redim_user_embedding_dict[task_name][dataset_pair[1]] = {}
        batch_size = 128
        source_items = list(item_embedding_dict[dataset_pair[0]].keys())
        target_items = list(item_embedding_dict[dataset_pair[1]].keys())
        source_users = list(user_embedding_dict[dataset_pair[0]].keys())
        target_users = list(user_embedding_dict[dataset_pair[1]].keys())
        with torch.no_grad():
            for i in tqdm(range(0, len(source_items), batch_size), desc='Reducing dim on source items'):
                batch_items = source_items[i:i+batch_size]
                batch_embs = torch.tensor(source_item_features[i:i+batch_size]).to(device).to(dtype)
                _, batch_redim_embs = model(batch_embs)
                batch_redim_embs = batch_redim_embs.detach().cpu().numpy()
                for j, item in enumerate(batch_items):
                    redim_item_embedding_dict[task_name][dataset_pair[0]][item] = batch_redim_embs[j]
            for i in tqdm(range(0, len(target_items), batch_size), desc='Reducing dim on target items'):
                batch_items = target_items[i:i+batch_size]
                batch_embs = torch.tensor(target_item_features[i:i+batch_size]).to(device).to(dtype)
                _, batch_redim_embs = model(batch_embs)
                batch_redim_embs = batch_redim_embs.detach().cpu().numpy()
                for j, item in enumerate(batch_items):
                    redim_item_embedding_dict[task_name][dataset_pair[1]][item] = batch_redim_embs[j]
            for i in tqdm(range(0, len(source_users), batch_size), desc='Reducing dim on source users'):
                batch_users = source_users[i:i+batch_size]
                batch_embs = torch.tensor(source_user_features[i:i+batch_size]).to(device).to(dtype)
                _, batch_redim_embs = model(batch_embs)
                batch_redim_embs = batch_redim_embs.detach().cpu().numpy()
                for j, user in enumerate(batch_users):
                    redim_user_embedding_dict[task_name][dataset_pair[0]][user] = batch_redim_embs[j]
            for i in tqdm(range(0, len(target_users), batch_size), desc='Reducing dim on target users'):
                batch_users = target_users[i:i+batch_size]
                batch_embs = torch.tensor(target_user_features[i:i+batch_size]).to(device).to(dtype)
                _, batch_redim_embs = model(batch_embs)
                batch_redim_embs = batch_redim_embs.detach().cpu().numpy()
                for j, user in enumerate(batch_users):
                    redim_user_embedding_dict[task_name][dataset_pair[1]][user] = batch_redim_embs[j]
        # redim_source_embeddings = np.stack(list(redim_item_embedding_dict[task_name][dataset_pair[0]].values()), axis=0)
        # redim_target_embeddings = np.stack(list(redim_item_embedding_dict[task_name][dataset_pair[1]].values()), axis=0)
        # itememb_visualize(redim_source_embeddings, redim_target_embeddings)
        # breakpoint()
        # redim_source_user_embeddings = np.stack(list(redim_user_embedding_dict[dataset_pair[0]].values()), axis=0)
        # redim_target_user_embeddings = np.stack(list(redim_user_embedding_dict[dataset_pair[1]].values()), axis=0)
        # itememb_visualize(redim_source_user_embeddings, redim_target_user_embeddings)
        # breakpoint()

    # breakpoint()

    for task_name in task_names.keys():
        dataset_pair = task_names[task_name]
        result_variable_dict[task_name] = {}
        user_info[task_name] = {}
        item_features[task_name] = {}
        for dataset in dataset_pair:
            print("User GMM weights learning stage for " + dataset + ", on task " + task_name)
            user_info[task_name][dataset] = build_userinfo(redim_user_embedding_dict[task_name][dataset], user_item_mapping[dataset], user_item_rating[dataset], redim_item_embedding_dict[task_name][dataset])
            item_features[task_name][dataset] = build_global_itemfeature(redim_item_embedding_dict[task_name][dataset])
    
            # 训练
            mu_k, sigma_k, sigma_inv_k, model, rating_predictor = train(user_info[task_name][dataset], item_features[task_name][dataset], task_name, dataset, Gaussian_Distributions_num[dataset])
            result_variable_dict[task_name][dataset] = {}
            result_variable_dict[task_name][dataset]["mu_k"] = mu_k
            result_variable_dict[task_name][dataset]["sigma_k"] = sigma_k
            result_variable_dict[task_name][dataset]["sigma_inv_k"] = sigma_inv_k
            result_variable_dict[task_name][dataset]["model"] = model
            result_variable_dict[task_name][dataset]["rating_predictor"] = rating_predictor

        print("Cross Domain Recommendation Stage for " + dataset_pair[0] + " and " + dataset_pair[1])
        source_domain = dataset_pair[0]
        target_domain = dataset_pair[1]
        task_label = source_domain + '_to_' + target_domain
        userlist_source = list(redim_user_embedding_dict[task_name][source_domain].keys())
        userlist_target = list(redim_user_embedding_dict[task_name][target_domain].keys())
        overlapped_user = list(set(userlist_source) & set(userlist_target))
        print("Num of Overlapped Users: ", len(overlapped_user))
        source_model = result_variable_dict[task_name][source_domain]["model"]
        target_rating_predictor = result_variable_dict[task_name][target_domain]["rating_predictor"]
        source_model.eval()
        target_rating_predictor.eval()
        ot_matrix = cal_ot_matrix(dataset_pair, result_variable_dict[task_name]) # (N, N')
        true_ratings_list = []
        predicted_ratings_list = []
        with torch.no_grad():
            for user in overlapped_user:
                source_embedding = torch.tensor(redim_user_embedding_dict[task_name][source_domain][user]).to(device).view(1, -1)
                w_u_source = source_model(source_embedding) # (1, N)
                w_u_target = w_u_source @ ot_matrix # (1, N')
                item_embeddings = torch.tensor(user_info[task_name][target_domain][user]["interactions"]).to(device).to(dtype).unsqueeze(0) # (1, num_items, feature_dim)
                predicted_ratings = predict_rating_with_weights(w_u_target, item_embeddings, result_variable_dict[task_name][target_domain]["mu_k"], result_variable_dict[task_name][target_domain]["sigma_inv_k"], target_rating_predictor) # (1, num_items)
                predicted_ratings = torch.clamp(predicted_ratings, min=1.0, max=5.0)
                true_ratings = torch.tensor(user_info[task_name][target_domain][user]["ratings"]).to(device).to(dtype).unsqueeze(0) # (1, num_items)
                predicted_ratings_list.append(predicted_ratings)
                true_ratings_list.append(true_ratings)
        predicted_ratings = torch.cat(predicted_ratings_list, dim=1) # (1, num_items)
        true_ratings = torch.cat(true_ratings_list, dim=1) # (1, num_items)
        mse = F.mse_loss(predicted_ratings, true_ratings)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(true_ratings - predicted_ratings))

        print(source_domain + "到" + target_domain + "的RMSE误差：", rmse.item())
        print(source_domain + "到" + target_domain + "的MAE误差：", mae.item())
        

if __name__=="__main__":
    main()