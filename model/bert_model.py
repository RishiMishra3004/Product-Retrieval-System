import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import os
import re

def clean_text(text):
    """
    Cleans text by removing HTML tags, special characters, and converting to lowercase.
    """
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def get_bert_embeddings(texts, model, tokenizer, batch_size=64):
    embeddings = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

def load_embeddings_and_data(embedding_file='Embeddings/Bert_Embeddings/bert_embeddings.npy', text_file='data/flipkart_com-ecommerce_sample.csv'):
    if os.path.exists(embedding_file):
        embeddings = np.load(embedding_file)  # Use numpy to load .npy files
    else:
        raise FileNotFoundError(f"Embedding file '{embedding_file}' not found.")
    
    data = pd.read_csv(text_file)
    text_columns = ['product_name', 'description', 'product_category_tree', 'product_url', 'retail_price', 'discounted_price']
    data[text_columns] = data[text_columns].fillna('')
    
    # Ensure correct data types
    data['retail_price'] = pd.to_numeric(data['retail_price'], errors='coerce')
    data['discounted_price'] = pd.to_numeric(data['discounted_price'], errors='coerce')
    
    data['text'] = data['product_name'] + ' ' + data['description'] + ' ' + data['product_category_tree']
    
    return embeddings, data

# Load BERT model and tokenizer once
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased').to(device)

def search_products_bert(query, top_n=10):
    query_clean = clean_text(query)
    # Load embeddings and data once
    embedding_file = 'Embeddings/Bert_Embeddings/bert_embeddings.npy'
    text_file = 'data/flipkart_com-ecommerce_sample.csv'

    query_embedding = get_bert_embeddings([query_clean], bert_model, tokenizer)
    embeddings, data = load_embeddings_and_data(embedding_file, text_file)

    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Retrieve the relevant columns based on top_indices
    results = data.iloc[top_indices][['product_name', 'product_url', 'retail_price', 'discounted_price']].reset_index(drop=True)
    return results
