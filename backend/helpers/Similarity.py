import numpy as np
import json
import re
from helpers.edit_distance import *


def jaccard(query, doc):
    intersection = query.intersection(doc)
    union = query.union(doc)
    return len(intersection) / len(union) if union else 0


def jaccard_mat(query, docs):
    return


def combined_jaccard_edit_distance(query, doc, jaccard_threshold=0.05):
    query_set = set(query.lower().split())
    doc_set = set(doc.lower().split())
    j = jaccard(query_set, doc_set)

    if j >= jaccard_threshold:
        edit_dist = edit_distance(query, doc)
        return edit_dist
    return float('inf')


def simpler_jaccard(query, doc, jaccard_threshold=0.05):
    query_set = set(query.lower())
    doc_set = set(doc.lower())
    j = jaccard(query_set, doc_set)

    if j >= jaccard_threshold:
        edit_dist = edit_distance(query, doc)
        return edit_dist
    return float('inf')


def removeLinksSpecials(comments_str):
    """
    Cleans up the input JSON string by removing HTML links and backslashes.
    """
    cleaned_string = re.sub(r'<a [^>]*>.*?</a>', '', comments_str)
    cleaned_string = cleaned_string.replace('\\', ' ')

    return cleaned_string


def avg_sentiment(comments_str):
    """
    Iterates over comment_dict_list, averaging sentiment over all comments.
    Returns 0.0 if any error occurs or if there are no valid sentiments to average.
    """
    comments_str = removeLinksSpecials(comments_str)
    if comments_str == "ERROR":
        return 0.0

    comments = json.loads(comments_str)
    if not comments:
        return 0.0

    total_sentiment = sum(comment.get('sentiment', 0.0)
                          for comment in comments)
    avg_sentiment = total_sentiment / len(comments)
    return np.round(avg_sentiment, 4)


def sentiment_similarity(sentiment1, sentiment2):
    # Assuming sentiments are scaled from -1 to 1
    return 1 - abs(sentiment1 - sentiment2)


def ted_talks_sim(talk1, talk2, doc_mat, index, w_transcript=1.0, w_summary=0.0, w_title=0.0, w_sentiment=0.0, w_speaker=0.0):
    """Returns a float giving the weighted similarity of 
       the two TED Talks.

    Params: {talk1 (str): Name of the first movie.
            talk2 (str): Name of the second movie.
            doc_mat (numpy.ndarray): Term-document matrix of movie transcripts, where 
                  each row represents a document (movie transcript) and each column represents a term.
            index (dict): Dictionary that maps movie names to the corresponding row index 
                  in the term-document matrix.}
            w_ (float): TED talk metadata types' (to be considered in
                  the similarity score) float weights. Weights sum to 1.0.
                  {"transcript" : _, "summary" : _, "title" : _, "sentiment" : _, "speaker" : _}

    Returns: Float (Cosine similarity of the two movie transcripts.)
    """
    # Make Helpers?

    # Transcripts: Cosine Similarity
    idx1 = index[talk1]
    idx2 = index[talk2]

    m1 = doc_mat[idx1]
    m2 = doc_mat[idx2]

    num = np.dot(m1, m2)

    denom = np.linalg.norm(m1) * np.linalg.norm(m2)

    cosine_sim = 0 if (denom == 0) else num/denom

    # Summary
    # Need a summary tf-idf matrix

    # Title
    # Need a title tf-idf matrix or edit-distnace?

    # Sentiment
    ##

    # Speaker
    # Check og_ds[_]['speaker']['name'] and og_ds[_]['speaker']['occupation']
    # og_ds[_]['speaker']['name']: Check if speakers are the same
    # og_ds[_]['speaker']['occupation']: Check edit distance

    return (cosine_sim * w_transcript) + (w_summary) + (w_title) + (w_sentiment) + (w_speaker)


def build_ted_talks_sims_cos(n, movie_index_to_name, input_doc_mat, movie_name_to_index):
    """Returns a ret_mat matrix of size (n,n) where for (i,j):
        [i,j] should be the similarity between the TED Talk with index i and the TED Talk with index j

    Note: You should set values on the diagonal to 1
    to indicate that all TED Talks are trivially perfectly similar to themselves.

    Params: {n: Integer, the number of movies
             movie_index_to_name: Dictionary, a dictionary that maps movie index to name
             input_doc_mat: Numpy Array, a numpy array that represents the document-term matrix
             movie_name_to_index: Dictionary, a dictionary that maps movie names to index
    Returns: Numpy Array 
    """
    ret_mat = np.zeros((n, n))

    for i in range(n):
        t1 = movie_index_to_name[i]
        for j in range(i+1):
            if i == j:
                ret_mat[i, j] = 1
            else:
                t2 = movie_index_to_name[j]
                score = ted_talks_sim(
                    t1, t2, input_doc_mat, movie_name_to_index)
                ret_mat[i, j] = ret_mat[j, i] = score

    return ret_mat

# THIS SHOULDNT WORK


def get_rankings(talk, matrix, index):
    """Returns a list of the most similar TED Talks to the
        inputted TED Talk.

    Parameters
    ----------
    talk : str (Length >= 2)
        TED Talk name 
    matrix : np.ndarray
        The term document matrix of the movie transcripts. input_doc_mat[i][j] is the tfidf
        of the movie i for the word j.
    index_to_vocab : dict
         A dictionary linking the index of a word (Key: int) to the actual word (Value: str). 
         Ex: {0: 'word_0', 1: 'word_1', .......}
    index : dict
         A dictionary linking the movie name (Key: str) to the movie index (Value: int). 
         Ex: {'movie_0': 0, 'movie_1': 1, .......}

    Returns
    -------
    list
        A list of the top k similar terms (in order) between the inputted movie transcripts
    """
    # Get movie index from movie name
    mov_idx = index[talk]

    # Get list of similarity scores for movie
    score_lst = matrix[mov_idx]
    # Pause. index[i] gets the index from the movie name (see line 105)
    cos_sim_mat = [(index[i], s) for i, s in enumerate(score_lst)]

    # Do not account for movie itself in ranking
    cos_sim_mat = cos_sim_mat[:mov_idx] + cos_sim_mat[mov_idx+1:]

    # Sort rankings by score
    cos_sim_mat = sorted(cos_sim_mat, key=lambda x: -x[1])

    return cos_sim_mat
