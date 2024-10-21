import pandas as pd
import pickle
import numpy as np

REVIEW_DROP = 0
RESTAURANTS_PATH = 'dataset/restaurants.csv'
USERS_PATH = 'dataset/users.csv'
REVIEWS_PATH = 'dataset/reviews.csv'

# Process users---------------------------------------------------------------------
print("Processing users")
users = pd.read_csv(USERS_PATH)
# Drop users with reviews count less than or equa REVIEW_DROP
users = users[users['review_count'] > REVIEW_DROP]
users['user_id'] = users['user_id'].astype('category')
users['user_id_num'] = users['user_id'].cat.codes
users = users[['user_id', 'user_id_num', 'review_count']]
user_id_to_num = dict(zip(users['user_id'], users['user_id_num']))
print(users)

# Process restaurants---------------------------------------------------------------
print("Processing restaurants")
restaurants = pd.read_csv(RESTAURANTS_PATH)
restaurants['business_id'] = restaurants['business_id'].astype('category')
restaurants['business_id_num'] = restaurants['business_id'].cat.codes
restaurants = restaurants[['business_id', 'business_id_num']]
rest_id_to_num = dict(zip(restaurants['business_id'], restaurants['business_id_num']))
print(restaurants)

# Process reviews--------------------------------------------------------------------
print("Processing reviews")
reviews = pd.read_csv(REVIEWS_PATH)
# Merge user and restaurant info
reviews = pd.merge(reviews, users, how='inner', on='user_id')
reviews = pd.merge(reviews, restaurants, how='inner', on='business_id')
# Drop id (keep id in number)
reviews = reviews.drop(columns='user_id')
reviews = reviews.drop(columns='business_id')
reviews = reviews.drop(columns='review_id')
# Process date column
reviews['date'] = pd.to_datetime(reviews['date'])
reviews['date'] = reviews['date'].astype('int64') // 10**9
# Keep only the numeric data columns
reviews = reviews.select_dtypes(include =[np.number])
print(reviews)

# Save data----------------------------------------------------------------------------
pickle.dump(user_id_to_num, open('./dataset/user_id_to_num.pkl', 'wb'))
pickle.dump(rest_id_to_num, open('./dataset/rest_id_to_num.pkl', 'wb'))
# Change type data of stars
np.save('./dataset/data.npy', reviews.values.astype(int))
# np.save('./dataset/data.npy', reviews.values)