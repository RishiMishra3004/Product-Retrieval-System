from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import re
import os

# Define the path for the embeddings file
sbert_embeddings_path = "Embeddings/Sbert_Embeddings/sbert_embeddings.pt"

# Load the dataset
data = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")

# Fill missing values in text fields with empty strings
text_columns = ['product_name', 'description', 'product_category_tree', 'brand']
data[text_columns] = data[text_columns].fillna('')

# Remove duplicates in place
data.drop_duplicates(inplace=True)

def clean_text(text):
    """
    Cleans text by removing HTML tags, special characters, and converting to lowercase.
    """
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Apply cleaning function
data['product_name'] = data['product_name'].apply(clean_text)
data['description'] = data['description'].apply(clean_text)
data['product_category_tree'] = data['product_category_tree'].apply(clean_text)

# Combine relevant text fields into a single column for vectorization
data['text'] = data['product_name'] + ' ' + data['description'] + ' ' + data['product_category_tree']

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load pre-trained SBERT model and move it to GPU if available
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model = sbert_model.to(device)

def load_sbert_embeddings(embeddings_path):
    """
    Load SBERT embeddings from the file.
    
    Parameters:
    - embeddings_path: Path to the SBERT embeddings file.
    
    Returns:
    - Tensor containing the SBERT embeddings.
    """
    embeddings = torch.load(embeddings_path, map_location=torch.device(device))
    return embeddings.to(device)  # Move embeddings to the same device as the model

def search_products(query, embeddings, top_n=10):
    """
    Search for products based on a query.

    Parameters:
    - query: The search query.
    - embeddings: Precomputed SBERT embeddings for the products.
    - top_n: Number of top similar products to return.

    Returns:
    - DataFrame with the top N most similar products.
    """
    # Convert the query to a vector using the same model used for product embeddings
    query_embedding = sbert_model.encode([query], convert_to_tensor=True, device=device)

    # Calculate cosine similarities between the query and product embeddings
    similarities = cosine_similarity(query_embedding.cpu(), embeddings.cpu()).flatten()

    # Get the indices of the top N most similar products
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Return the top N most similar products
    return data.iloc[top_indices][['product_name', 'retail_price', 'overall_rating', 'product_url']].reset_index(drop=True) 


def search_products_sbert(query, sbert_embeddings):
    sbert_embeddings = load_sbert_embeddings(sbert_embeddings_path)
    results = search_products(query, sbert_embeddings, top_n=10)
    return results

