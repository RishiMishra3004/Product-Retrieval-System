from transformers import BertTokenizer, BertModel
import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os

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

# Load the BERT model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased').to(device)

def get_bert_embeddings(texts, model, tokenizer, batch_size=64):
    embeddings = []
    num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)

    for start in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Generating BERT embeddings"):
        end = start + batch_size
        batch_texts = texts[start:end]
        # Tokenize texts and move tensors to GPU if available
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Take the mean of the token embeddings (CLS token can be used too)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)  # Stack arrays vertically

def generate_bert_embeddings():
    if os.path.exists('Embeddings/Bert_Embeddings/bert_embeddings.npy'):
        print("Embeddings already generated and saved.")
        return 
    texts = data['text'].tolist()  # Ensure texts are in a list format
    bert_embeddings = get_bert_embeddings(texts, bert_model, tokenizer)
    # Save embeddings to a file
    np.save('Embeddings/Bert_Embeddings/bert_embeddings.npy', bert_embeddings)