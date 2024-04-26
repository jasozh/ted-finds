import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

# Build Sim Matrix using SVD


def build_sim_matrix(documents, k=100):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=.7, min_df=75)
    td_matrix = vectorizer.fit_transform(documents)

    # u, s, v_trans = svds(td_matrix, k=k)

    docs_compressed, s, words_compressed = svds(td_matrix, k=int(k/2))
    words_compressed = words_compressed.transpose()
    docs_compressed_normed = normalize(docs_compressed)

    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}

    categories = {}
    category_names = [
    "Creativity & Play",
    "Artificial Intelligence & Expression",
    "Science & Environmental",
    "Identity & Society",
    "Knowledge & Economy",
    "Health & Relationships",
    "Education & Awareness",
    "Artistic Pursuits",
    "Childhood & Play",
    "Cultural & Health Exchange",
    "Environmental Biology",
    "Narrative & Nourishment",
    "Health & Design",
    "Global Health & Development",
    "Microbiology & Nutrition",
    "Health & Science",
    "Technology & Imagination",
    "Sustainability & Consumption",
    "Global Energy & Education",
    "Marine Ecology & Conservation",
    "Design & Biology",
    "Planetary Health & Geology",
    "Data Science & Diversity",
    "Language & Planetary Sciences",
    "Technology & Nutrition",
    "Psychology & Physiology",
    "Language & Medicine",
    "Food Systems & Sociology",
    "African Art & Health",
    "Business & Data Analysis",
    "Water Resources & Linguistics",
    "Financial Technology & Robotics",
    "Sleep & Linguistics",
    "Urban Data Analysis & Climate",
    "Robotics & Oceanography",
    "Humor & Environmental Science",
    "Urban Life & Environmental Studies",
    "Neuroscience & Environmental Studies",
    "Data Analysis & Childhood Education",
    "Cosmology & Health Education",
    "Brain Studies & Urban Development",
    "Neurology & Climate Change",
    "Gender Studies & Energy",
    "Music Therapy & Environmental Science",
    "Humor & Earth Sciences",
    "Music & Oceanography",
    "Neuroscience & Health",
    "Psychology & Music Therapy",
    "Humor & Psychology",
    "Knowledge & Communication"
    ]

    for i in range(50):
        dimension_col = words_compressed[:,i].squeeze()
        categories[i] = category_names[i]


    #td_matrix_np = td_matrix.transpose().toarray()
    #td_matrix_np = normalize(td_matrix_np)

    sim_matrix = docs_compressed_normed.dot(docs_compressed_normed.T)
    doc_category_scores = docs_compressed#doc_category_scores = docs_compressed_normed.dot(words_compressed.T)
    #print(sim_matrix.shape, doc_category_scores.shape)
    return sim_matrix, categories, doc_category_scores


    # td_matrix_np = td_matrix.transpose().toarray()
    # td_matrix_np = normalize(td_matrix_np)

    sim_matrix = docs_compressed_normed.dot(docs_compressed_normed.T)
    return sim_matrix


def get_top_k(title, title_to_idx, sim_matrix, k=10):
    idx = title_to_idx[title]
    top_k_indices = sim_matrix[idx].argsort()[::-1][1:k]
    top_k_values = sim_matrix[idx][top_k_indices]
    return top_k_indices, top_k_values

def get_doc_category_scores(docs, categories, dc_matrix, docname_to_idx, idx_to_categories):

    idxs = [docname_to_idx[name] for name in docs]
    c_idxs = [c for c in categories]

    docs = []

    for doc in idxs:
        d = {}
        for c_idx in c_idxs:
            d[idx_to_categories[str(c_idx)]] = dc_matrix[doc, c_idx]
        docs.append(d)

    return docs

def get_categories(query, dc_matrix, names_to_idx):
    idx = names_to_idx[query]
    x = np.argsort(dc_matrix[idx])[:10]
    return np.argsort(dc_matrix[idx])[:10]

def get_top_k_talks(title, title_to_idx, idx_to_title, sim_matrix, dc_matrix, idx_to_categories, categories_to_idx, k=10):
    top_k_indices, top_k_values = get_top_k(title, title_to_idx, sim_matrix, k)
    #tops =  [(idx_to_title[str(idx)], score) for (idx, score) in zip(top_k_indices, top_k_values)]
    docs = [idx_to_title[str(idx)] for idx in top_k_indices]
    categories = get_categories(title, dc_matrix, title_to_idx)
    top_dc_scores = get_doc_category_scores(docs, categories, dc_matrix, title_to_idx, idx_to_categories)
    #query_scores = [{idx_to_categories[str(cat)] : dc_matrix[title_to_idx[title], cat]} for cat in categories]
    query_scores = {}
    for cat in categories:
        category = idx_to_categories[str(cat)]
        query_scores[category] = dc_matrix[title_to_idx[title], int(cat)]
    return [(idx_to_title[str(idx)], (score, cats)) for (idx, score, cats) in zip(top_k_indices, top_k_values, top_dc_scores)], query_scores


def prepare_data():
    # Preprocess Data
    with open('../init.json', 'r') as json_file:
        data = json.load(json_file)
        talks = pd.json_normalize(data)

    # Get transcripts in one place
    documents = []
    for idx, row in talks.iterrows():
        documents.append((row['title'], row['transcript']))

    idx_to_title = {}
    title_to_idx = {}
    for idx, (title, _) in enumerate(documents):
        idx_to_title[idx] = title
        title_to_idx[title] = idx

    sim_matrix, categories, doc_category_scores = build_sim_matrix([transcript[1] for transcript in documents])
    sim_matrix = np.round(sim_matrix, 4)

    with open("idx_to_docnames", 'w') as json_file:
        json.dump(idx_to_title, json_file)
    with open("docname_to_idx", 'w') as json_file:
        json.dump(title_to_idx, json_file)

    with open("categories", "w") as json_file:
        json.dump(categories, json_file)

    categories_inv = {value: key for key, value in categories.items()}

    with open("categories_inv", "w") as json_file:
        json.dump(categories_inv, json_file)


    #np.save('doc_category_scores', doc_category_scores)


    chunk_size = len(sim_matrix) // 6
    chunks = [sim_matrix[i:i+chunk_size]
              for i in range(0, len(sim_matrix), chunk_size)]
    for i, chunk in enumerate(chunks):
        filename = f'chunk_{i}.npy'
        np.save(filename, chunk)

    chunks_2 = [doc_category_scores[i:i+chunk_size]
              for i in range(0, len(doc_category_scores), chunk_size)]
    for i, chunk in enumerate(chunks_2):
        filename = f'dc_chunk_{i}.npy'
        np.save(filename, chunk)

#prepare_data()
    

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

    loaded_chunks = []
    for i in range(6):
        filename = f'chunk_{i}.npy'
        loaded_chunk = np.load(filename)
        loaded_chunks.append(loaded_chunk)

    dc_matrix = np.concatenate(loaded_chunks)

    with open("categories", 'r') as json_file:
        idx_to_categories = json.load(json_file)

    top_k_talks = get_top_k_talks(query, docname_to_idx, inv, sim_matrix, 10)
    titles = [title for title, _ in top_k_talks]
    categories = get_categories(query, dc_matrix, idx_to_categories)
    dc_scores = get_doc_category_scores(titles, categories)

    zipped = zip(top_k_talks, dc_scores)
    
    return zipped

if __name__ == '__main__':
    prepare_data()
