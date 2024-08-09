import numpy as np
import re
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords

data = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")

# Fill missing values in text fields with empty strings
text_columns = ['product_name', 'description', 'product_category_tree', 'brand']
data[text_columns] = data[text_columns].fillna('')

# Remove duplicates
data.drop_duplicates()

# Define English stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens.
    """
    return [word for word in tokens if word not in stop_words]

data['product_name'] = data['product_name'].apply(clean_text)
data['description'] = data['description'].apply(clean_text)
data['product_category_tree'] = data['product_category_tree'].apply(clean_text)

# Combine relevant text fields into a single column for vectorization
data['text'] = data['product_name'] + ' ' + data['description'] + ' ' + data['product_category_tree']

# Tokenize the text
data['tokens'] = data['text'].apply(word_tokenize)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=data['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Function to get Word2Vec vectors for a product
def get_word2vec_vector(text):
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    
    vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    if len(vectors) == 0:
        # Handle case where no tokens are in the vocabulary
        return np.zeros(word2vec_model.vector_size)
    vector = np.mean(vectors, axis=0)
    return vector

def generate_word2vec_embeddings():
    # Apply the function to get Word2Vec vectors with progress display
    word2vec_vectors = np.array([get_word2vec_vector(text) for text in tqdm(data['text'])])

    # Save the Word2Vec model to disk
    word2vec_model.save("Embeddings/Word2Vec_Embeddings/word2vec_model.model")

    # Save the Word2Vec vectors to a file
    np.save("Embeddings/Word2Vec_Embeddings/word2vec_vectors.npy", word2vec_vectors)

    # Display the shape of the Word2Vec vectors
    # print("Word2Vec vectors shape:", word2vec_vectors.shape)