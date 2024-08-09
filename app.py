import streamlit as st
import pandas as pd
import re
# Import functions from model files
from model.heuristic_search import heuristic_search
from model.tfidf_model import search_products_tfidf
from model.word2vec_model import search_products_word2vec
from model.sbert_model import search_products_sbert
# from model.bert_model import search_products_bert
from multilingual_query import translate_to_english

# Load the data
data = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")  # Adjust the path as necessary

# Streamlit app
st.title("Product Search App")

# Search bar
query = st.text_input("Search for products:")

# Dropdown menu for model selection
model_selection = st.selectbox(
    "Select the model to use for search",
    ["Heuristic", "TF-IDF", "Word2Vec", "SBERT"]
)

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Initialize results dictionary
results = {}

# Create two columns for buttons
col1, col2 = st.columns(2)

# Place buttons in columns
with col1:
    search_button = st.button("SEARCH")
with col2:
    clear_button = st.button("CLEAR")
    
# Handle button actions
if search_button:
    if query:
        query = clean_text(translate_to_english(query))
        
        # Heuristic search
        heuristic_results = heuristic_search(query)
        results["Heuristic"] = heuristic_results
        
        # TF-IDF
        tfidf_results = search_products_tfidf(query, data)
        results["TF-IDF"] = tfidf_results

        # Word2Vec
        word2vec_results = search_products_word2vec(query, data)
        results["Word2Vec"] = word2vec_results

        # SBERT
        sbert_results = search_products_sbert(query, data)
        results["SBERT"] = sbert_results

        # BERT (commented out)
        # bert_results = search_products_bert(query)
        # results["BERT"] = bert_results
        
        # Display selected model results at the top
        if model_selection in results:
            st.subheader(f"{model_selection} RESULTS")
            st.write(results[model_selection])
        
        # Display other models' results below
        for model in ["Heuristic", "TF-IDF", "Word2Vec", "SBERT", "BERT"]:
            if model != model_selection and model in results:
                if(model != "Heuristic"):
                    st.subheader("ML-BASED MODEL")
                st.subheader(f"{model} Search Results")
                st.write(results[model])
elif clear_button:
    query = ""
    results = {}
    st.write("Results cleared.")
