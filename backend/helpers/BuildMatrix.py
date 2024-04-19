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
        asort = np.argsort(-dimension_col)
        print([index_to_word[i] for i in asort[:5]])
        print()
        categories[i] = category_names[i]


    #td_matrix_np = td_matrix.transpose().toarray()
    #td_matrix_np = normalize(td_matrix_np)

    sim_matrix = docs_compressed_normed.dot(docs_compressed_normed.T)
    doc_category_scores = docs_compressed_normed.dot(words_compressed.T)
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


def get_top_k_talks(title, title_to_idx, idx_to_title, sim_matrix, k=10):
    top_k_indices, top_k_values = get_top_k(title, title_to_idx, sim_matrix, k)
    return [(idx_to_title[str(idx)], score) for (idx, score) in zip(top_k_indices, top_k_values)]


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

    #np.save('doc_category_scores', doc_category_scores)


    chunk_size = len(sim_matrix) // 6
    chunks = [sim_matrix[i:i+chunk_size]
              for i in range(0, len(sim_matrix), chunk_size)]
    for i, chunk in enumerate(chunks):
        filename = f'chunk_{i}.npy'
        np.save(filename, chunk)

    chunks_2 = [doc_category_scores[i:i+chunk_size]
              for i in range(0, len(doc_category_scores), chunk_size)]
    for i, chunk in enumerate(chunks):
        filename = f'dc_chunk_{i}.npy'
        np.save(filename, chunk)

    
def get_doc_category_scores(docs, categories):
    # Load Similarity Matrix
    loaded_chunks = []
    for i in range(6):
        filename = f'chunk_{i}.npy'
        loaded_chunk = np.load(filename)
        loaded_chunks.append(loaded_chunk)

    dc_matrix = np.concatenate(loaded_chunks)

    with open("categories", 'r') as json_file:
        categories_to_idx = json.load(json_file)

    with open("docname_to_idx", 'r') as json_file:
        docname_to_idx = json.load(json_file)

    idxs = [docname_to_idx[name] for name in docs]

    docs = {}
    
    for doc in idxs:
        docs[doc] = {}

    for c in categories:
        c_idx = categories_to_idx[c]
        for doc in idxs:
            docs[doc][c_idx] = dc_matrix[doc, c_idx]

    return docs

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

    return get_top_k_talks(query, docname_to_idx, inv, sim_matrix, 10)

#prepare_data()