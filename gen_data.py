import pandas as pd
import gzip
import json
import os
from tqdm import tqdm
import re
import html

# 解压文件
def parse(path):
    g = gzip.open(path, 'rb')#以二进制形式打开路径的文件
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}#创建空字典
    for d in tqdm(parse(path), desc="Reading file"):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def clean_text(text):
    # 转换HTML实体字符
    text = html.unescape(text)
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除竖线分隔符
    text = re.sub(r'\|', '', text)
    # 去除列表符号，如 [] 和 []
    text = re.sub(r'\[.*?\]', '', text)
    # 去除额外的引号
    text = re.sub(r"['\"]", '', text)
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_data(dataset, rewrite=True):
    cur_path = os.path.dirname(__file__)
    reviewData = dataset + ".json.gz"
    review_path = os.path.join(cur_path, "dataset", dataset, reviewData)
    metaData = "meta_" + reviewData
    meta_path = os.path.join(cur_path, "dataset", dataset, metaData)
    review_json = dataset + ".json"
    meta_json = "meta_" + review_json
    review_json_path = os.path.join(cur_path, "dataset", dataset, review_json)
    meta_json_path = os.path.join(cur_path, "dataset", dataset, meta_json)

    # 获得DataFrame类型的数据
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    DF_review = getDF(review_path)
    DF_meta = getDF(meta_path)

    DF_meta = DF_meta.drop_duplicates(subset=['asin'])
    DF_meta_dict = DF_meta.set_index('asin').to_dict(orient='index')

    # 根据交互数据构建图
    # 构建结点信息
    user_item_mapping = DF_review.groupby('reviewerID')['asin'].apply(list).to_dict()
    user_item_mapping = {user: [item for item in items] for user, items in user_item_mapping.items() if len(items) >= 10}
    print("user数量：",len(list(user_item_mapping.keys())))
    
    user_ids = list(user_item_mapping.keys())
    user_item_preference_dict = {}
    for user in tqdm(user_ids, desc="Processing Users"):
        user_item_preference_dict[user] = {}
        for row in DF_review.loc[DF_review['reviewerID'] == user].itertuples():
            item_data = DF_meta_dict.get(row.asin, {})
            title = item_data.get('title', 'No Title Information')
            description = ' '.join(item_data.get('description', ['No Description Information.']))
            if row.reviewText:
                review = str(row.reviewText)
            else:
                review = "No Review Information."
            if row.overall:
                rating = float(row.overall)
            else:
                rating = 3.0
            if row.unixReviewTime:
                review_time = int(row.unixReviewTime)
            else:
                review_time = 0
            item_review = 'Item Title: ' + title + '; Item Description: ' + description + '; User\'s Review: ' + review
            item_review = clean_text(item_review)
            user_item_preference_dict[user][str(row.asin)] = {}
            user_item_preference_dict[user][str(row.asin)]["review"] = item_review
            user_item_preference_dict[user][str(row.asin)]["rating"] = rating
            user_item_preference_dict[user][str(row.asin)]["unixReviewTime"] = review_time

    item_ids = list(set(item for sublist in user_item_mapping.values() for item in sublist))
    item_description_dict = {}
    for item in tqdm(item_ids, desc="Processing Items"):
        item_data = DF_meta_dict.get(item, {})
        title = item_data.get('title', 'No Title Information')
        description = ' '.join(item_data.get('description', ['No Description Information']))
        overall_description = 'Item Title: ' + title + '; Item Description: ' + description
        overall_description = clean_text(overall_description)
        item_description_dict[item] = overall_description

    data = {"user_interaction":user_item_preference_dict, "item_description":item_description_dict}
    with open('./dataset/'+ dataset + '.json', 'w') as j:
        json.dump(data, j)

if __name__ == '__main__':
    # get_data('All_Beauty', rewrite=True)
    # get_data('AMAZON_FASHION', rewrite=True)
    # get_data('Gift_Cards', rewrite=True)
    # get_data('Magazine_Subscriptions', rewrite=True)
    # get_data('Movies_and_TV', rewrite=True)
    get_data('Books', rewrite=True)
    # get_data('Cell_Phones_and_Accessories', rewrite=True)
    # get_data('Electronics', rewrite=True)
    # get_data('Digital_Music', rewrite=True)
    # get_data('Video_Games', rewrite=True)
    # get_data('Kindle_Store', rewrite=True)