import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'static/data/init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    df = pd.read_json(file)
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

@app.route("/")
def home():
    return render_template('home.html', title="Home", autocomplete=titles)

@app.route("/results")
def results():
    search_query = request.args.get('query')
    data = [{
        'title': ['Presentation 1'],
        'page_url': ['https://example.com/1'],
        'likes': [1000],
        'recorded_date': ['2024-01-01'],
        'speakers': ['Speaker A'],
        'topics': ['Topic X'],
        'views': [5000],
        'summary': ['Summary of Ted Talk']
    }]
    data = pd.DataFrame(data)
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