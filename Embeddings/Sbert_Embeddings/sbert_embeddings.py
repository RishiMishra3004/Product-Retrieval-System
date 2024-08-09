from sentence_transformers import SentenceTransformer
import torch
import re
import pandas as pd
import os
from tqdm import tqdm

# Define the path for the embeddings file
sbert_embeddings_path = "Embeddings/SBERT_Embeddings/sbert_embeddings.pt"

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

def generate_and_save_sbert_embeddings(data, model, device, embeddings_path, batch_size=32):
    """
    Generate SBERT embeddings for the product data and save them to a file.
    
    Parameters:
    - data: Pandas DataFrame containing the product data.
    - model: SBERT model for encoding.
    - device: Device ('cuda' or 'cpu') for computation.
    - embeddings_path: Path to save the SBERT embeddings.
    - batch_size: Number of samples to process in each batch.
    """
    if not os.path.exists(embeddings_path):
        # Generate SBERT embeddings for each product with progress display
        print("Generating SBERT embeddings for the first time.")
        texts = data['text'].tolist()
        sbert_embeddings = []

        # Process texts in batches
        for start in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            embeddings = model.encode(batch_texts, convert_to_tensor=True, device=device)
            sbert_embeddings.append(embeddings)  # Collect tensors

        # Stack the list of tensors into a single tensor
        sbert_embeddings = torch.cat(sbert_embeddings, dim=0)

        # Save the SBERT embeddings to a file
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)  # Create directories if they don't exist
        torch.save(sbert_embeddings, embeddings_path)
        print(f"SBERT embeddings saved to {embeddings_path}")
    else:
        print("SBERT embeddings already exist. No need to generate again.")

# Generate and save SBERT embeddings if they do not exist
def generate_sbert_embeddings():
    generate_and_save_sbert_embeddings(data, sbert_model, device, sbert_embeddings_path)
    # Display the shape of the SBERT embeddings
    # print("SBERT embeddings shape:", sbert_embeddings.shape)

