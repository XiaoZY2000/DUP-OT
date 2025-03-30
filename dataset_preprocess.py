import numpy as np
from tqdm import tqdm

def cal_user_emb(user_item_mapping, item_embedding_dict):
    user_embedding_dict = {}
    for user in user_item_mapping.keys():
        interacted_item_embeddings = []
        for item in user_item_mapping[user]:
            item_embedding = item_embedding_dict[item]
            interacted_item_embeddings.append(item_embedding)
        user_embedding = np.mean(np.stack(interacted_item_embeddings), axis=0)
        user_embedding_dict[user] = user_embedding
    return user_embedding_dict

def build_userinfo(user_embedding_dict, user_item_mapping, user_item_rating, item_embedding_dict):
    # 目标字典：{user_1: {"embedding": user_embedding, "interactions": interacted_item_embeddings, "ratings": ratings}}
    user_info = {}
    for user in user_embedding_dict.keys():
        user_info[user] = {}
        user_info[user]["embedding"] = user_embedding_dict[user]
        interactions = []
        for item in user_item_mapping[user]:
            interactions.append(item_embedding_dict[item])
        user_info[user]["interactions"] = np.stack(interactions)
        user_info[user]["ratings"] = user_item_rating[user]

    return user_info

def build_global_itemfeature(item_embedding_dict):
    item_embeddings = []
    for item in item_embedding_dict.keys():
        item_embeddings.append(item_embedding_dict[item])
    item_features = np.stack(item_embeddings)

    return item_features