import numpy as np

def jaccard(query, doc):
    return 

def jaccard_mat(query, docs):
    return

def ted_talks_cos_sim(talk1, talk2, doc_mat, index):
    idx1 = index[talk1]
    idx2 = index[talk2]

    m1 = doc_mat[idx1]
    m2 = doc_mat[idx2]

    num = np.dot(m1, m2)

    denom = np.linalg.norm(m1) * np.linalg.norm(m2)

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
          score = ted_talks_cos_sim(t1, t2, input_doc_mat, movie_name_to_index)
          ret_mat[i, j] = ret_mat[j, i] = score

    return ret_mat

def get_rankings(talk, matrix, index):
    # Get movie index from movie name
    mov_idx = index[talk]

    # Get list of similarity scores for movie
    score_lst = matrix[mov_idx]
    cos_sim_mat = [(index[i], s) for i,s in enumerate(score_lst)]

    # Do not account for movie itself in ranking
    cos_sim_mat = cos_sim_mat[:mov_idx] + cos_sim_mat[mov_idx+1:]

    # Sort rankings by score
    cos_sim_mat = sorted(cos_sim_mat, key=lambda x: -x[1])

    return cos_sim_mat
    