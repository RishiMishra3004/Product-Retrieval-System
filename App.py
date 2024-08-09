import streamlit as st
import pandas as pd
import re
# Import functions from model files
from model.heuristic_search import heuristic_search
from model.tfidf_model import search_products_tfidf
from model.word2vec_model import search_products_word2vec
from model.sbert_model import search_products_sbert
# from model.bert_model import search_products_bert

# Load the data
data = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")  # Adjust the path as necessary

# Streamlit app
st.title("Product Search App")

# Search bar
query = st.text_input("Search for products:")

# Dropdown menu for model selection
model_selection = st.selectbox(
    "Select the model to use for search",
    ["Heuristic", "TF-IDF", "Word2Vec", "SBERT", "BERT"]
)

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# query = "tshirts"
if query:
    query = clean_text(query)
    # Heuristic search
    st.subheader("Heuristic Search Results")
    heuristic_results = heuristic_search(query)
    print(heuristic_results)
    st.write(heuristic_results)

    # Machine Learning based search
    st.subheader("ML Model-Based Search Results")

    # TF-IDF
    st.subheader("TF-IDF Search Results")
    tfidf_results = search_products_tfidf(query, data)
    st.write(tfidf_results)

    # Word2Vec
    st.subheader("Word2Vec Search Results")
    word2vec_results = search_products_word2vec(query, data)
    st.write(word2vec_results)

    # BERT
    st.subheader("BERT Search Results")
    sbert_results = search_products_sbert(query, data)
    st.write(sbert_results)

    # # BERT
    # st.subheader("BERT Search Results")
    # bert_results = search_products_bert(query)
    # st.write(bert_results)
