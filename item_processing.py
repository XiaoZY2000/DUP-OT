from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import ast
import tiktoken
import json
import pickle
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from collections import defaultdict

encoder_name='sentence-t5-base'
encoder = SentenceTransformer(encoder_name)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
max_tokens = tokenizer.model_max_length

def get_chunk_embedding(tokens):
    stride = int(max_tokens/2)
    chunks = []

    for i in range(0, len(tokens), stride):
        chunk = tokens[i: i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    
    chunk_embeddings = encoder.encode(chunks, convert_to_numpy=True)
    
    return np.mean(chunk_embeddings, axis=0)

def text_preprocess(text):
    processed_text = text.replace("Item Title: ","")
    processed_text = processed_text.replace("; Item Description: ",":")
    return processed_text

def get_item_embedding_from_reviews(review_list):
    all_sentences = []
    sentence_counts = []

    # 1. 对所有 review 分句，统一收集
    for review in review_list:
        review = text_preprocess(review)
        sentences = sent_tokenize(review)
        all_sentences.extend(sentences)
        sentence_counts.append(len(sentences))

    # 2. 批量编码所有句子（提高效率）
    sent_embeddings = encoder.encode(
        all_sentences,
        batch_size=64,
        convert_to_tensor=False,  # 如果后续用 numpy 操作建议保持为 False
        show_progress_bar=False
    )

    # 3. 重新组合每条 review 的句子嵌入，求平均
    review_vectors = []
    idx = 0
    for count in sentence_counts:
        review_embed = np.mean(sent_embeddings[idx:idx+count], axis=0)
        review_vectors.append(review_embed)
        idx += count

    # 4. 求所有 review 的平均，得到 item 向量
    item_embedding = np.mean(review_vectors, axis=0)
    return item_embedding

def extract_user_and_item_reviews(nested_dict):
    user_reviews = {}
    item_reviews = defaultdict(list)

    for user_id, item_dict in nested_dict.items():
        user_review_list = []
        for item_id, item_info in item_dict.items():
            review_text = item_info.get("review", "")
            if review_text:
                user_review_list.append(review_text)
                item_reviews[item_id].append(review_text)
        user_reviews[user_id] = user_review_list

    return user_reviews, dict(item_reviews)

def embed(item_reviews_dict):
    item_embedding_dict = {}
    for item in tqdm(item_reviews_dict.keys(), desc="Embedding Items/Users"):
        embedding = get_item_embedding_from_reviews(item_reviews_dict[item])
        item_embedding_dict[item] = embedding
    return item_embedding_dict

def embed_all_items_fast(item_reviews_dict):
    all_sentences = []
    sent_map = []  # 每个句子属于哪个 item 和哪个 review（item_id, review_index）

    for item_id, reviews in tqdm(item_reviews_dict.items(), desc='Dividing Sentences'):
        for review_index, review in enumerate(reviews):
            review = text_preprocess(review)
            sentences = sent_tokenize(review)
            for sent in sentences:
                all_sentences.append(sent)
                sent_map.append((item_id, review_index))

    # 编码所有句子
    sent_embeddings = encoder.encode(
        all_sentences,
        batch_size=128,
        convert_to_tensor=False,
        show_progress_bar=True
    )

    # 构建 {item_id: [review_embed, ...]}
    from collections import defaultdict
    item_review_embeddings = defaultdict(lambda: defaultdict(list))

    for i, (item_id, review_index) in enumerate(sent_map):
        item_review_embeddings[item_id][review_index].append(sent_embeddings[i])

    # 计算每条 review 的平均，再计算 item 的平均
    item_embedding_dict = {}
    for item_id, review_dict in item_review_embeddings.items():
        review_embeds = [
            np.mean(emb_list, axis=0)
            for emb_list in review_dict.values()
        ]
        item_embed = np.mean(review_embeds, axis=0)
        item_embedding_dict[item_id] = item_embed

    return item_embedding_dict

def itememb_visualize(source_item_features, target_item_features):
    label_1 = np.zeros(len(source_item_features), dtype=int)
    label_2 = np.ones(len(target_item_features), dtype=int)
    labels = np.concatenate([label_1, label_2])
    data_points = np.concatenate([source_item_features, target_item_features], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_data = tsne.fit_transform(data_points)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced_data[:,0], reduced_data[:,1], c=labels, cmap='viridis', s=10)
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(scatter, label="Labels")
    plt.savefig("Item Embedding.png")

def split_random_rows(arr, sample_size):
    num_rows = arr.shape[0]

    idx = np.random.choice(num_rows, size=sample_size, replace=False)
    return arr[idx]