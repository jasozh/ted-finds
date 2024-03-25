import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from helpers.edit_distance import *
import pandas as pd
import numpy as np
import time
import helpers.BuildMatrix as bm

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

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
    df = pd.read_json(file).head(500)
    titles = df["title"]

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

def autocomplete_filter(search_query: str) -> list[tuple[str, int]]:
    """
    Filters the list of suggested titles based on edit distance
    """
    start_time = time.time()
    q = search_query.lower()

    # Find smallest 5 edit distance
    n = titles.size
    edit_distance_arr = np.zeros(n) # (i, val) = (df index, edit distance to q)
    for i in range(n):
        d = titles.iloc[i].lower()
        edit_distance_arr[i] = edit_distance(q, d)
    top_5_indices = np.argsort(edit_distance_arr)[:5]

    # Return as list
    result = [(titles.iloc[i], edit_distance_arr[i]) for i in top_5_indices]

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
    else:
        search_query = ""
        autocomplete = [(title, "") for title in titles[:5]]
    return render_template('home.html', title="Home", query=search_query, autocomplete=autocomplete)

def get_top_10_for_query(query):
    p1 = os.path.join(current_directory, 'helpers/docname_to_idx')
    p2 = os.path.join(current_directory, 'helpers/idx_to_docnames')
    p3 = os.path.join(current_directory, 'helpers/cosine_similarity_matrix.npy')
    with open(p1, 'r') as json_file:    
        docname_to_idx = json.load(json_file)
    with open(p2, 'r') as json_file:    
        inv = json.load(json_file)
    matrix = np.load(p3)
    return bm.get_rankings(query, matrix, docname_to_idx, inv)[:10]

df['speakers'] = df['speakers'].apply(lambda x: json.loads(x))
df['topics'] = df['topics'].apply(lambda x: json.loads(x))

@app.route("/results")
def results():
    search_query = request.args.get('q')

    results = get_top_10_for_query(search_query)

    titles = [result[0] for result in results]

    data = df[df["title"].isin(titles)]
    #data = {'results': df.to_dict(orient='records')}

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

    return render_template('results.html', title="Results", data=data)

@app.route("/video")
def video():
    return render_template('video.html', title="Video")

@app.route("/example")
def example():
    return render_template('example.html', title="Example")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)