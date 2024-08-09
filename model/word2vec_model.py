import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load the Word2Vec model
word2vec_model = Word2Vec.load("Embeddings/Word2Vec_Embeddings/word2vec_model.model")

# Load the Word2Vec vectors
word2vec_vectors = np.load("Embeddings/Word2Vec_Embeddings/word2vec_vectors.npy")

# Load the original dataset
data = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")

# Define English stopwords
stop_words = set(stopwords.words('english'))

# Ensure data is consistent
# assert len(data) == len(word2vec_vectors), "Mismatch between data and vectors!"
def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens.
    """
    return [word for word in tokens if word not in stop_words]

def get_word2vec_vector(text):
    """
    Convert a text into its Word2Vec vector representation by averaging the vectors
    of its tokens that are present in the Word2Vec model's vocabulary.
    """
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    
    vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    if len(vectors) == 0:
        # Handle case where no tokens are in the vocabulary
        return np.zeros(word2vec_model.vector_size)
    vector = np.mean(vectors, axis=0)
    return vector

def search_products_word2vec(query, data, top_n=10):
    """
    Search for the top N products that are most similar to the query using Word2Vec embeddings.

    Args:
    query (str): The search query.
    data (pd.DataFrame): The dataframe containing product information.
    top_n (int): The number of top similar products to return.

    Returns:
    pd.DataFrame: A dataframe containing the top N similar products with their details.
    """
    # Clean the query

    # Get the Word2Vec vector for the query
    query_vector = get_word2vec_vector(query)

    # Calculate cosine similarity between the query vector and the product vectors
    similarities = cosine_similarity([query_vector], word2vec_vectors).flatten()

    # Get the top N most similar products
    top_indices = similarities.argsort()[-top_n:][::-1]
    result_df = data.iloc[top_indices][['product_name', 'product_url', 'retail_price', 'discounted_price']].reset_index(drop=True)

    # Add similarity scores to the DataFrame
    result_df['similarity_score'] = similarities[top_indices]

    return result_df
