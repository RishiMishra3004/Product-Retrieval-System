from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import pandas as pd
import joblib

data = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")

# Fill missing values in text fields with empty strings
text_columns = ['product_name', 'description', 'product_category_tree', 'brand']
data[text_columns] = data[text_columns].fillna('')

# Remove duplicates
data.drop_duplicates()

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

data['product_name'] = data['product_name'].apply(clean_text)
data['description'] = data['description'].apply(clean_text)
data['product_category_tree'] = data['product_category_tree'].apply(clean_text)

# Combine relevant text fields into a single column for vectorization
data['text'] = data['product_name'] + ' ' + data['description'] + ' ' + data['product_category_tree']

def generate_tfidf_embeddings():
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    # Fit and transform the text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])

    # Save the TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, 'Embeddings/Tfidf_Embedding/tfidf_vectorizer.pkl')

    # Save the TF-IDF matrix
    joblib.dump(tfidf_matrix, 'Embeddings/Tfidf_Embedding/tfidf_matrix.pkl')

# Display the shape of the TF-IDF matrix
# print("TF-IDF matrix shape:", tfidf_matrix.shape)
