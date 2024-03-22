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
    tokens = {}
    for transcript in transcripts:
        for token in tokenize_transcript(tokenize, transcript[1]):
            tokens[token] = tokens.get(token, 0) + 1
    return tokens

def create_ranked_good_types(
    tokenize_method, tokenize_transcript_method, input_transcripts, input_good_types):
    words_combined = []
    for transcript in input_transcripts:
        words = tokenize_transcript_method(tokenize_method, transcript[1])
        words_combined += words

    tup_list = []
    total_len = len(words_combined)
    
    for word in input_good_types:
        word_freq = round(words_combined.count(word) / total_len, 5)
        tup_list.append((word, word_freq))
    sorted_tups = sorted(tup_list, key=lambda x : x[1], reverse=True)
    
    return sorted_tups

def create_word_occurrence_matrix(
    tokenize_method,
    input_transcripts,
    input_good_types):
    
    speaker_indices = {speaker[2]: idx for idx, speaker in enumerate(input_transcripts)}
    freqs = np.zeros((len(input_transcripts), len(input_good_types)))

    word_idx = {}
    for idx, (word, freq) in enumerate(input_good_types):
        word_idx[idx] = word

    for idx, transcript, _ in input_transcripts:
        tokens = tokenize_method(transcript)
        arr = np.zeros(len(input_good_types))
        for word_index in word_idx:
            good_word = word_idx[word_index]
            arr[word_index] = tokens.count(good_word)
        freqs[idx] += arr[:]

    return speaker_indices, freqs

def df_to_word_occ_mat(df):
    list_of_tuples = []

    for idx, row in df.iterrows():
        list_of_tuples.append((idx, row['transcript'], row['title']))

    counts = types_counts(list_of_tuples)

    filtered_counts = {}
    for word in counts:
        if (counts[word] > 100):
            filtered_counts[word] = counts[word]

    good_types = create_ranked_good_types(tokenize, tokenize_transcript, list_of_tuples, filtered_counts)
    docname_to_idx, word_occ_mat = create_word_occurrence_matrix(tokenize, list_of_tuples, good_types)

    return docname_to_idx, word_occ_mat

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the TED Talk transcript data as a Pandas dataframe
    """
    with open('backend/init.json', 'r') as json_file:
        data = json.load(json_file)
        df = pd.json_normalize(data)
        df = df.head(200)
    docname_to_idx, word_occ_mat = df_to_word_occ_mat(df)
    return docname_to_idx, word_occ_mat

docname_to_idx, word_occ_mat = load_data("")