import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from helpers.edit_distance import *
from helpers.Similarity import combined_jaccard_edit_distance, simpler_jaccard
import pandas as pd
import numpy as np
import time
import helpers.BuildMatrix as bm

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    """
    Creates a Pandas DataFrame with the following keys for each TED Talk:
        "_id"
        "duration"
        "event"
        "likes"
        "page_url"
        "published_date"
        "recorded_date"
        "related_videos"
        "speakers"
        "subtitle_languages"
        "summary"
        "title"
        "topics"
        "transcript"
        "views"
        "youtube_video_code"
        "comments"
    """
    df = pd.read_json(file)
    df = df[df['transcript'] != '']
    titles = df["title"]

    df['speakers'] = df['speakers'].apply(lambda x: json.loads(x))
    df['topics'] = df['topics'].apply(lambda x: json.loads(x))

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas


def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df,
                         left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower(
    ).str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


def get_top_10_for_query(query):
    p1 = os.path.join(current_directory, 'helpers/docname_to_idx')
    p2 = os.path.join(current_directory, 'helpers/idx_to_docnames')
    # p3 = os.path.join(current_directory,
    #                  'helpers/cosine_similarity_matrix.npy')

    loaded_chunks = []
    for i in range(6):
        filename = os.path.join(current_directory, f'helpers/chunk_{i}.npy')
        loaded_chunk = np.load(filename)
        loaded_chunks.append(loaded_chunk)
    sim_matrix = np.concatenate(loaded_chunks)

    with open(p1, 'r') as json_file:
        docname_to_idx = json.load(json_file)
    with open(p2, 'r') as json_file:
        inv = json.load(json_file)

    dc_loaded_chunks = []
    for i in range(6):
        filename = os.path.join(current_directory, f'helpers/dc_chunk_{i}.npy')
        dc_loaded_chunk = np.load(filename)
        dc_loaded_chunks.append(dc_loaded_chunk)

    dc_matrix = np.concatenate(dc_loaded_chunks)

    with open(os.path.join(current_directory, 'helpers/categories'), 'r') as json_file:
        idx_to_categories = json.load(json_file)

    with open(os.path.join(current_directory, 'helpers/categories_inv'), 'r') as json_file:
        categories_to_idx = json.load(json_file)

    # matrix = np.load(p3)
    return bm.get_top_k_talks(query, docname_to_idx, inv, sim_matrix, dc_matrix, idx_to_categories, categories_to_idx, 10)


def autocomplete_filter(search_query: str) -> list[tuple[str, int]]:
    """
    Filters the list of suggested titles based on edit distance
    """
    start_time = time.time()
    q = search_query.lower()

    # Find smallest 5 edit distance
    n = titles.size
    # (i, val) = (df index, edit distance to q)
    sim_arr = np.zeros(n)
    for i in range(n):
        d = titles.iloc[i].lower()
        sim_arr[i] = combined_jaccard_edit_distance(q, d)
    top_5_indices = np.argsort(sim_arr)[:5]

    # Return as list
    result = [(titles.iloc[i], sim_arr[i])
              for i in top_5_indices if sim_arr[i] != float('inf')]

    # Measure performance
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The operation took {elapsed_time:.4f} seconds.")

    return result


def simple_filter(search_query: str) -> list[tuple[str, int]]:
    """
    Filters the list of suggested titles based on edit distance
    """
    start_time = time.time()
    q = search_query.lower()

    # Find smallest 5 edit distance
    n = titles.size
    # (i, val) = (df index, edit distance to q)
    sim_arr = np.zeros(n)
    for i in range(n):
        d = titles.iloc[i].lower()
        sim_arr[i] = simpler_jaccard(q, d)
    top_5_indices = np.argsort(sim_arr)[:5]

    # Return as list
    result = [(titles.iloc[i], sim_arr[i])
              for i in top_5_indices if sim_arr[i] != float('inf')]

    # Measure performance
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The operation took {elapsed_time:.4f} seconds.")

    return result


@app.route("/")
@app.route("/search")
def home():
    search_query = request.args.get("q")
    if search_query:
        autocomplete = autocomplete_filter(search_query)
        if (len(autocomplete) < 5):
            autocomplete = simple_filter(search_query)
    else:
        search_query = ""
        autocomplete = [(title, "") for title in titles[:5]]
    return render_template('home.html', title="Home", query=search_query, autocomplete=autocomplete)


@app.route("/results")
def results():
    search_query = request.args.get('q')

    results, query_cat_scores = get_top_10_for_query(search_query)

    titles = [result[0] for result in results]
    # print(len(titles))
    titles_scores_dict = dict(results)
    # print(titles_scores_dict)

    data = df[df["title"].isin(titles)]

    # Create new column
    data["cosine_similarity"] = [-1 for _ in range(len(data))]
    data["category_scores"] = [{}] * len(data)
    # print(titles_scores_dict)
    for i, video in data.iterrows():
        title = video["title"]
        data.at[i, "cosine_similarity"] = round(
            titles_scores_dict[title][0]*100, 2)
        # data.loc[i, "category_scores"] = titles_scores_dict[title][1]
        data.at[i, "category_scores"] = titles_scores_dict[title][1]

    # dictionary of the form {1597: -0.033, 3356: -0.0952, 5614: -0.3411, ... } is stored in titles_scores_dict[title][1]

    # Sort data by cosine_similarity in descending order
    sorted_data = data.sort_values(by="cosine_similarity", ascending=False)

    print(sorted_data.iloc[0]['category_scores'])
    print(query_cat_scores)

    # data = [{
    #     'title': ['Presentation 1'],
    #     'page_url': ['https://example.com/1'],
    #     'likes': [1000],
    #     'recorded_date': ['2024-01-01'],
    #     'speakers': ['Speaker A'],
    #     'topics': ['Topic X'],
    #     'views': [5000],
    #     'summary': ['Summary of Ted Talk']
    # }]

    return render_template('results.html', title="Results", search_query=search_query, data=sorted_data, query_scores=query_cat_scores)


@app.route("/video")
def video():
    video_id = int(request.args.get('w'))
    data = df[df["_id"] == video_id].iloc[0]

    # Cosine Similarity
    results, _ = get_top_10_for_query(data.title)

    titles = [result[0] for result in results]
    # print(len(titles))
    titles_scores_dict = dict(results)
    # print(titles_scores_dict)

    data = df[df["title"].isin(titles)]
    print(data)

    # Create new column
    data["cosine_similarity"] = [-1 for _ in range(len(data))]
    data["category_scores"] = [{}] * len(data)
    # print(titles_scores_dict)
    for i, video in data.iterrows():
        title = video["title"]
        data.at[i, "cosine_similarity"] = round(
            titles_scores_dict[title][0]*100, 2)
        # data.loc[i, "category_scores"] = titles_scores_dict[title][1]
        data.at[i, "category_scores"] = titles_scores_dict[title][1]

    # dictionary of the form {1597: -0.033, 3356: -0.0952, 5614: -0.3411, ... } is stored in titles_scores_dict[title][1]

    # Sort data by cosine_similarity in descending order
    sorted_related_videos = data.sort_values(by="cosine_similarity", ascending=False)

    # Get comments
    try:
        comments = json.loads(df[df["_id"] == video_id].iloc[0]['comments'])
        #comments = data['comments']
        print("Loaded")
    except:
        comments = {}
        print("No comments")

    positive_comments = sorted([
        comments[i]
        for i in range(len(comments))
        if comments[i]["sentiment"] > 0
    ], reverse=True, key=lambda comment: comment["comment_likes"])[:3]

    negative_comments = sorted([
        comments[i]
        for i in range(len(comments))
        if comments[i]["sentiment"] < 0
    ], reverse=True, key=lambda comment: comment["comment_likes"])[:3]

    # positive_comments = [
    #     "Positive Comment 1 from YouTube...",
    #     "Positive Comment 2 from YouTube...",
    #     "Positive Comment 3 from YouTube...",
    # ]
    # negative_comments = [
    #     "Negative Comment 1 from YouTube...",
    #     "Negative Comment 2 from YouTube...",
    #     "Negative Comment 3 from YouTube...",
    # ]
    return render_template('video.html', title="Video", data=data, related_videos=sorted_related_videos, positive_comments=positive_comments, negative_comments=negative_comments)


@ app.route("/example")
def example():
    return render_template('example.html', title="Example")


@ app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
