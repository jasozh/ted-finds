import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from helpers.Similarity import avg_sentiment, sentiment_similarity

# Build Sim Matrix using SVD

def build_sim_matrix(documents, k=100):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=.7, min_df=75)
    td_matrix = vectorizer.fit_transform(documents)

    # u, s, v_trans = svds(td_matrix, k=k)

    docs_compressed, _, _ = svds(td_matrix, k=int(k/2))
    # words_compressed = words_compressed.transpose()
    docs_compressed_normed = normalize(docs_compressed)

    # td_matrix_np = td_matrix.transpose().toarray()
    # td_matrix_np = normalize(td_matrix_np)

    sim_matrix = docs_compressed_normed.dot(docs_compressed_normed.T)
    return sim_matrix

def get_top_k(title, title_to_idx, idx_to_sentiments, sim_matrix, k=10, sentiment_sim=0.1):
    idx = title_to_idx[title]
    query_sentiment = idx_to_sentiments[str(idx)]
    sentiment_scores = np.array([sentiment_similarity(query_sentiment, s) for s in idx_to_sentiments.values()])
    combined_scores = (1.0 - sentiment_sim) * (sim_matrix[idx]) + sentiment_sim * sentiment_scores
    top_k_indices = combined_scores.argsort()[::-1][1:k+1]
    top_k_values = combined_scores[top_k_indices]
    return top_k_indices, top_k_values


def get_top_k_talks(title, title_to_idx, idx_to_title, idx_to_sentiments, sim_matrix, k=10):
    top_k_indices, top_k_values = get_top_k(title, title_to_idx, idx_to_sentiments, sim_matrix, k)
    return [(idx_to_title[str(idx)], score) for (idx, score) in zip(top_k_indices, top_k_values)]


def prepare_data():
    # Preprocess Data
    with open('../init.json', 'r') as json_file:
        data = json.load(json_file)
        talks = pd.json_normalize(data)
        talks = talks[talks['transcript'] != '']

    # Get transcripts in one place
    documents = []
    idx_to_sentiments = {}
    for idx, row in talks.iterrows():
        idx_to_sentiments[idx] = avg_sentiment(row['comments'])
        documents.append((row['title'], row['transcript']))

    idx_to_title = {}
    title_to_idx = {}
    for idx, (title, _) in enumerate(documents):
        idx_to_title[idx] = title
        title_to_idx[title] = idx

    sim_matrix = build_sim_matrix([transcript[1] for transcript in documents])
    sim_matrix = np.round(sim_matrix, 4)

    with open("idx_to_docnames", 'w') as json_file:
        json.dump(idx_to_title, json_file)
    with open("docname_to_idx", 'w') as json_file:
        json.dump(title_to_idx, json_file)
    with open("idx_to_sentiments", 'w') as json_file:
        json.dump(idx_to_sentiments, json_file)

    chunk_size = len(sim_matrix) // 6
    chunks = [sim_matrix[i:i+chunk_size]
              for i in range(0, len(sim_matrix), chunk_size)]
    for i, chunk in enumerate(chunks):
        filename = f'chunk_{i}.npy'
        np.save(filename, chunk)


def get_top_10_for_query(query):
    # Load Similarity Matrix
    loaded_chunks = []
    for i in range(6):
        filename = f'chunk_{i}.npy'
        loaded_chunk = np.load(filename)
        loaded_chunks.append(loaded_chunk)
    sim_matrix = np.concatenate(loaded_chunks)

    # Load docname_to_idx, idx_to_docname
    with open("docname_to_idx", 'r') as json_file:
        docname_to_idx = json.load(json_file)
    with open("idx_to_docnames", 'r') as json_file:
        inv = json.load(json_file)
    with open("idx_to_sentiments", 'r') as json_file:
        idx_to_sentiments = json.load(json_file)

    return get_top_k_talks(query, docname_to_idx, inv, idx_to_sentiments, sim_matrix, 10)

if __name__ == '__main__':
    prepare_data()