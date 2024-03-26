import numpy as np
import pandas as pd
import re
import json

def tokenize(text):
    text = text.lower()
    return re.findall("[a-z]+", text)

def tokenize_transcript(tokenize_method, transcript):
    return tokenize_method(transcript)

def types_counts(transcripts):
    print("Started type counts")
    tokens = {}
    for transcript in transcripts:
        for token in tokenize_transcript(tokenize, transcript[1]):
            tokens[token] = tokens.get(token, 0) + 1
    print("Done Type Counts")
    return tokens

def create_ranked_good_types(
    tokenize_method, tokenize_transcript_method, input_transcripts, input_good_types):
    words_combined = []
    for transcript in input_transcripts:
        words = tokenize_transcript_method(tokenize_method, transcript[1])
        words_combined += words

    tup_list = []
    total_len = len(words_combined)

    print("Started ranking")
    
    for word in input_good_types:
        word_freq = round(words_combined.count(word) / total_len, 5)
        tup_list.append((word, word_freq))
    #sorted_tups = sorted(tup_list, key=lambda x : x[1], reverse=True)
        
    print("Done rankings")
    
    return tup_list#sorted_tups

def create_word_occurrence_matrix(
    input_transcripts,
    input_good_types):

    print("Creating WOC mat")
    
    speaker_indices = {speaker[2]: idx for idx, speaker in enumerate(input_transcripts)}
    freqs = np.zeros((len(input_transcripts), len(input_good_types)))

    word_idx = {}
    for idx, word in enumerate(input_good_types):
        word_idx[idx] = word

    for idx, _, _, tokenized in input_transcripts:
        tokens = tokenized
        arr = np.zeros(len(input_good_types))
        for word_index in word_idx:
            good_word = word_idx[word_index]
            arr[word_index] = tokens.count(good_word)
        freqs[idx] += arr[:]
        print(f"Done {idx}")

    return speaker_indices, freqs

def df_to_word_occ_mat(df):
    list_of_tuples = []

    for idx, row in df.iterrows():
        list_of_tuples.append((idx, row['summary'], row['title'], tokenize_transcript(tokenize, row['transcript'])))

    counts = types_counts(list_of_tuples)

    filtered_counts = {}
    for word in counts:
        if (counts[word] < 5000):
            filtered_counts[word] = counts[word]
    print(len(filtered_counts))

    #good_types = create_ranked_good_types(tokenize, tokenize_transcript, list_of_tuples, filtered_counts)
    docname_to_idx, word_occ_mat = create_word_occurrence_matrix(list_of_tuples, filtered_counts)

    return docname_to_idx, word_occ_mat

def get_sim(talk1, talk2, input_doc_mat, input_movie_name_to_index):

    idx1 = input_movie_name_to_index[talk1]
    idx2 = input_movie_name_to_index[talk2]

    m1 = input_doc_mat[idx1]
    m2 = input_doc_mat[idx2]

    num = np.dot(m1, m2)

    denom = np.linalg.norm(m1) * np.linalg.norm(m2)

    if denom == 0:
        return 0
    return num/denom

def build_ted_talks_sims_cos(n, movie_index_to_name, input_doc_mat, movie_name_to_index):

    ret_mat = np.zeros((n, n))

    for i in range(n):
      t1 = movie_index_to_name[i]
      for j in range(i+1):
        if i == j:
          ret_mat[i, j] = 1
        else:
          t2 = movie_index_to_name[j]
          score = get_sim(t1, t2, input_doc_mat, movie_name_to_index)
          ret_mat[i, j] = ret_mat[j, i] = score

    return ret_mat

def get_rankings(talk, matrix, index, inv):
    # Get movie index from movie name
    mov_idx = index[talk]

    # Get list of similarity scores for movie
    score_lst = matrix[mov_idx]
    cos_sim_mat = [(inv[str(i)], s) for i,s in enumerate(score_lst)]

    # Do not account for movie itself in ranking
    cos_sim_mat = cos_sim_mat[:mov_idx] + cos_sim_mat[mov_idx+1:]

    # Sort rankings by score
    cos_sim_mat = sorted(cos_sim_mat, key=lambda x: -x[1])

    return cos_sim_mat

def invert_dict(d):
    inverted_dict = {}
    for item in d:
        inverted_dict[d[item]] = item
    return inverted_dict

def load_data():
    """
    Loads the TED Talk transcript data as a Pandas dataframe
    """
    with open('../init.json', 'r') as json_file:
        data = json.load(json_file)
        df = pd.json_normalize(data)
    docname_to_idx, word_occ_mat = df_to_word_occ_mat(df)
    return docname_to_idx, word_occ_mat

def build_similarity_matrix():
    docname_to_idx, word_occ_mat = load_data()
    print("Done Loading Data")
    inv = invert_dict(docname_to_idx)
    matrix = build_ted_talks_sims_cos(
        n=len(word_occ_mat), 
        movie_index_to_name=inv, 
        input_doc_mat=word_occ_mat, 
        movie_name_to_index=docname_to_idx)
    print("Done Building Matrix")
    np.save("cosine_similarity_matrix", matrix)
    with open("idx_to_docnames", 'w') as json_file:
        json.dump(inv, json_file)
    with open("docname_to_idx", 'w') as json_file:
        json.dump(docname_to_idx, json_file)

def get_top_10_for_query(query):
    with open("docname_to_idx", 'r') as json_file:    
        docname_to_idx = json.load(json_file)
    with open("idx_to_docnames", 'r') as json_file:    
        inv = json.load(json_file)
    # Load the four parts from the .npy files
    part1 = np.load('part1.npy')
    part2 = np.load('part2.npy')
    part3 = np.load('part3.npy')
    part4 = np.load('part4.npy')

    # Concatenate the parts to reconstruct the original matrix
    top = np.concatenate((part1, part2), axis=1)
    bottom = np.concatenate((part3, part4), axis=1)
    matrix = np.concatenate((top, bottom), axis=0)
    return get_rankings(query, matrix, docname_to_idx, inv)[:10]