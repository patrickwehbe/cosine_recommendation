import pymongo
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['my_database']
collection = db['my_collection']

# Retrieve products from MongoDB
products = pd.DataFrame(list(collection.find()))

# Create a pivot table of user ratings
ratings = pd.pivot_table(products, values='rating', index='user_id', columns='product_id')

# Compute the similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(ratings.fillna(0))

# Create a function to get similar users
def get_similar_users(user_id, ratings, similarity_matrix):
    user_ratings = ratings.loc[user_id].fillna(0)
    similarity = similarity_matrix[user_id]
    weighted_sum = similarity.dot(ratings.fillna(0))
    weighted_sum /= similarity.sum()
    return pd.DataFrame({'product_id': ratings.columns, 'score': weighted_sum}).sort_values(by='score', ascending=False)

# Get recommendations for a user
user_id = 123
similar_users = get_similar_users(user_id, ratings, similarity_matrix)
recommendations = products[products['user_id'].isin(similar_users.head(10)['user_id'])].groupby('product_id').mean().sort_values(by='rating', ascending=False).head(10)
