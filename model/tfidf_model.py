from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('Embeddings/Tfidf_Embedding/tfidf_vectorizer.pkl')

# Load the TF-IDF matrix
tfidf_matrix = joblib.load('Embeddings/Tfidf_Embedding/tfidf_matrix.pkl')

def search_products_tfidf(query, data, top_n=10):
    query_tfidf = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return data.iloc[top_indices][['product_name', 'product_url', 'retail_price', 'discounted_price']].reset_index(drop=True)  