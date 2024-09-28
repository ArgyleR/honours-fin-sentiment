import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def apply_vader_ranking(text_series, ids, text_dates, top_n=None):
    """
    Rank texts by VADER sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text)['compound'] for text in text_series]
    ranked_indices = np.argsort(scores)[::-1][:top_n] if top_n is not None else np.argsort(scores)[::-1]
    
    text_series = [text_series[i] for i in ranked_indices]
    ids = [ids[i] for i in ranked_indices]
    text_dates = [text_dates[i] for i in ranked_indices]
    
    return text_series, ids, text_dates

def apply_clustering_ranking(text_series, ids, text_dates, top_n=None):
    """
    Rank texts based on clustering (KMeans) and select representative texts.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_series)

    n_clusters = min(len(text_series), 5)  # Avoid too many clusters for a small number of texts
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)

    representative_texts = []
    representative_ids = []
    representative_dates = []

    # Get one representative from each cluster
    for cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        representative_idx = cluster_indices[0]  # Pick the first item in the cluster
        representative_texts.append(text_series[representative_idx])
        representative_ids.append(ids[representative_idx])
        representative_dates.append(text_dates[representative_idx])

    return representative_texts[:top_n], representative_ids[:top_n], representative_dates[:top_n]

def apply_embedding_diversity_ranking(text_series, ids, text_dates, model, top_n=None, batch_size=8):
    """
    Rank texts based on embedding diversity using SentenceTransformer for faster embeddings.
    - Uses SentenceTransformer for efficient embedding generation.
    - Supports GPU acceleration.
    """
    
    # Generate embeddings for all texts using SentenceTransformer (it handles batching internally)
    embeddings = model.encode(text_series, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    
    # Compute cosine similarity between the embeddings
    similarity_matrix = cosine_similarity(embeddings)

    selected_indices = []
    current_idx = 0

    # Select least similar texts iteratively
    while len(selected_indices) < len(text_series):
        selected_indices.append(current_idx)
        similarities = similarity_matrix[current_idx]
        similarities[selected_indices] = 1.0  # Mark selected indices as highly similar to avoid reselecting them
        current_idx = np.argmin(similarities)  # Select the least similar remaining text

    selected_indices = selected_indices[:top_n] if top_n is not None else selected_indices

    text_series = [text_series[i] for i in selected_indices]
    ids = [ids[i] for i in selected_indices]
    text_dates = [text_dates[i] for i in selected_indices]

    return text_series, ids, text_dates

def apply_tfidf_ranking(text_series, ids, text_dates, top_n=None):
    """
    Rank texts by TF-IDF scores and select top_n texts.
    """
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform(text_series)
        
        if tfidf_matrix.shape[1] == 0:  # Check if TF-IDF matrix is empty
            raise ValueError("Empty TF-IDF matrix.")
        
        # Sum the TF-IDF scores for each document and flatten it to 1D
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get the indices of the top_n texts
        top_indices = np.argsort(tfidf_scores)[-top_n:]

        # Select the top_n texts, ids, and dates
        text_series = [text_series[i] for i in top_indices]
        ids = [ids[i] for i in top_indices]
        text_dates = [text_dates[i] for i in top_indices]
    
    except ValueError as e:
        # Handle cases where the TF-IDF matrix is empty
        print(f"Skipping window due to TF-IDF error: {e}")
        return text_series, ids, text_dates

    return text_series, ids, text_dates