import pandas as pd
import json
import os
from tqdm import tqdm
import re
import html

def clean_text(text):
    # Remove HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'\|', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'\[.*?\]', '', text)
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r"['\"]", '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_data(dataset, rewrite=True):
    cur_path = os.path.dirname(__file__)
    reviewData = dataset + ".jsonl"
    review_path = os.path.join(cur_path, "dataset", dataset, reviewData)
    metaData = "meta_" + reviewData
    meta_path = os.path.join(cur_path, "dataset", dataset, metaData)

    # Get the data
    DF_review = pd.read_json(review_path, lines=True)
    print("Review Data Loaded")
    DF_meta = pd.read_json(meta_path, lines=True)
    print("Meta Data Loaded")
    DF_review_dict = (
    DF_review.groupby('user_id')
      .apply(lambda g: g.drop(columns='user_id').to_dict(orient='records'))
      .to_dict()
)
    print("User Data Transformed to Dictionary")

    DF_meta = DF_meta.drop_duplicates(subset=['parent_asin'])
    DF_meta_dict = DF_meta.set_index('parent_asin').to_dict(orient='index') # Turn into a dictionary for faster access

    print("Item Data Transformed to Dictionary")
    
    interaction_dict = {}

    for user_id, reviews in tqdm(DF_review_dict.items(), desc="Processing Reviews"):
        for review in reviews:
            item_id = review.get('parent_asin')
            if item_id is not None:
                if not (review['rating'] and review['text']): # Filter out empty reviews
                    continue
                interaction_dict[user_id] = {}
                interaction_dict[user_id][item_id] = {'review': clean_text(review.get('title', '') + ': ' + review['text']),
                                                      'rating': float(review['rating'])}
    
    # Item ids after filtering
    item_ids = set([item for user_interactions in interaction_dict.values() for item in user_interactions.keys()])

    item_metadata_dict = {}
    for item in tqdm(item_ids, desc="Processing Items"):
        item_data = DF_meta_dict.get(item, {})
        if not item_data:
            continue
        title = item_data.get('title', '')
        description = ' '.join(item_data.get('description', []))
        overall_description = title + ': ' + description
        overall_description = clean_text(overall_description)
        images_all = item_data.get('images', [])
        images = []
        for image in images_all:
            images.append(image.get('large', ''))
        item_metadata_dict[item] = {}
        item_metadata_dict[item]['description'] = overall_description
        item_metadata_dict[item]['images'] = images

    data = {"interaction":interaction_dict, "metadata":item_metadata_dict}
    with open('./dataset/'+ dataset + '/data.json', 'w') as j:
        json.dump(data, j)

if __name__ == '__main__':
    # get_data('All_Beauty', rewrite=True)
    # get_data('AMAZON_FASHION', rewrite=True)
    # get_data('Gift_Cards', rewrite=True)
    # get_data('Magazine_Subscriptions', rewrite=True)
    get_data('Movies_and_TV', rewrite=True)
    # get_data('Books', rewrite=True)
    # get_data('Cell_Phones_and_Accessories', rewrite=True)
    # get_data('Electronics', rewrite=True)
    # get_data('Digital_Music', rewrite=True)
    get_data('Video_Games', rewrite=True)
    get_data('Kindle_Store', rewrite=True)
    # get_data('CDs_and_Vinyl', rewrite=True)