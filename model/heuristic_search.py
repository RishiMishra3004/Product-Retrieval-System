import pandas as pd
from collections import defaultdict
import re

data = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")

# Convert non-string values to empty strings
data['description'] = data['description'].astype(str)
# Define a set of stop words
stop_words = set([
    'a', 'an', 'the', 'is', 'in', 'on', 'and', 'or', 'to', 'for', 'of', 'with', 'by', 'from', 'at', 'as', 'it', 'that', 'which', 'are', 'was', 'were'
])

def preprocess_text(text, stop_words):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)  # Remove punctuation except hyphens
        text = text.replace('>>', ' ')
        words = text.split()
        words = [word for word in words if word not in stop_words]  # Remove stop words
        text = ' '.join(words)
    else:
        text = ''  # Handle non-string values
    return text


# Fill missing values with an empty string or some placeholder
data['description'] = data['description'].fillna('')

# Apply preprocessing with stop word removal
data['processed_description'] = data['description'].apply(lambda x: preprocess_text(x, stop_words))

# Apply preprocessing
data['processed_category'] = data['product_category_tree'].apply(lambda x: preprocess_text(x, stop_words))

def build_index(descriptions, categories):
    index = defaultdict(list)
    for i, (desc, cat) in enumerate(zip(descriptions, categories)):
        if isinstance(desc, str):
            words = desc.split()
            for word in words:
                index[word].append(i)
        if isinstance(cat, str):
            words = cat.split()
            for word in words:
                index[word].append(i)
    return index

# Build the index
index = build_index(data['processed_description'], data['processed_category'])


def search(query, index, descriptions, categories, stop_words):
    query = preprocess_text(query, stop_words)
    query_words = query.split()
    if not query_words:
        return []

    matched_indices = set()
    for word in query_words:
        if word in index:
            matched_indices.update(index[word])

    scores = defaultdict(int)
    for idx in matched_indices:
        product_description = descriptions[idx]
        product_category = categories[idx]

        desc_word_count = sum(product_description.count(word) for word in query_words)
        cat_word_count = sum(product_category.count(word) for word in query_words)

        scores[idx] += cat_word_count * 10
        scores[idx] += desc_word_count

        if all(word in product_description for word in query_words):
            scores[idx] += 500

    ranked_indices = sorted(scores, key=scores.get, reverse=True)
    return ranked_indices

# Function to retrieve and display top 10 product details
def display_top_results(indices, data, top_n=10):
    # Retrieve rows from data corresponding to the indices
    results_df = data.loc[indices]
    
    # Select relevant columns for display
    display_columns = ['product_name', 'product_url', 'retail_price', 'discounted_price']
    
    # Filter out only relevant columns for better readability
    results_df = results_df[display_columns]
    results_df.reset_index(drop=True, inplace=True)  # Reset index to sequential
    
    # Limit the results to top_n rows
    return results_df.head(top_n)


def heuristic_search(query, top_n=10):
    """
    Perform a heuristic search on the dataset based on the query.

    Parameters:
    - query (str): The search term.
    - data (pd.DataFrame): The dataset containing product information.
    - top_n (int): The number of top results to return.

    Returns:
    - pd.DataFrame: The top N search results.
    """
    
    # Simple heuristic search: find products containing the query in their text
    results = search(query, index, data['processed_description'], data['processed_category'], stop_words)

    # Display the top 10 results in a DataFrame
    top_results_df = display_top_results(results, data, top_n)
    return top_results_df
