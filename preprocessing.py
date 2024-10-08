import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import ast

# unix datetime
base = pd.Timestamp("1970-01-01")
CHUNK_SIZE = 1000000
REVIEW_DROP = 0
RESTAURANTS_PATH = 'dataset/processed_rest.csv'
REVIEWS_PATH = 'dataset/reviews.csv'
USERS_PATH = 'dataset/processed_users.csv'

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).long().to(device)

# Load data files
# reviews = get_reviews()
def load_data(train_percent, val_percent, test_percent):
    print("Reading users")
    users = pd.read_csv(USERS_PATH)
    users = users[users['review_count'] > REVIEW_DROP]
    users['user_id'] = users['user_id'].astype('category')
    users['user_id_num'] = users['user_id'].cat.codes
    users = users[['user_id', 'user_id_num', 'review_count']]
    user_id_to_num = dict(zip(users['user_id'], users['user_id_num']))
    print(users)

    print("Reading businesses")
    restaurants = pd.read_csv(RESTAURANTS_PATH)
    restaurants['business_id'] = restaurants['business_id'].astype('category')
    restaurants['business_id_num'] = restaurants['business_id'].cat.codes
    restaurants = restaurants[['business_id', 'business_id_num']]
    rest_id_to_num = dict(zip(restaurants['business_id'], restaurants['business_id_num']))
    print(restaurants)

    print("Reading reviews")
    reviews = pd.read_csv(REVIEWS_PATH)

    reviews = pd.merge(reviews, users, how='inner', on='user_id')
    reviews = reviews.drop(columns='user_id')
    reviews = pd.merge(reviews, restaurants, how='inner', on='business_id')
    reviews = reviews.drop(columns='business_id')
    print("REVIEWS.HEAD() -------------------------------------------------------------------")
    print(reviews.head())
    reviews = reviews.drop(columns=reviews.columns[0], axis=1)
    print("REVIEWS.DROP() -------------------------------------------------------------------")
    print(reviews.head())
    reviews = reviews.select_dtypes(include =[np.number])
    print("reviews.select_dtypes() -------------------------------------------------------------------")
    
    print(reviews.head())

    pickle.dump(user_id_to_num, open('./dataset/user_id_to_num.pkl', 'wb'))
    pickle.dump(rest_id_to_num, open('./dataset/rest_id_to_num.pkl', 'wb'))
    np.save('./dataset/data.npy', reviews.values)

    training = reviews.sample(frac=train_percent)

    left = reviews.drop(training.index)
    validation = left.sample(frac=val_percent / (val_percent + test_percent))

    test = left.drop(validation.index)

    print("loaded")

    return df_to_tensor(training), df_to_tensor(validation), df_to_tensor(test), user_id_to_num, rest_id_to_num


if __name__ == "__main__":
    train, val, test, user, rest = load_data(0.6, 0.3, 0.1)
    print("TRAIN ----------------------------------------------")
    print(train.shape)
    print("VAL ----------------------------------------------")
    print(val.shape)
    print("TEST ----------------------------------------------")
    print(test.shape)