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
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    episodes_df = pd.DataFrame(data['episodes'])
    reviews_df = pd.DataFrame(data['reviews'])

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
    return render_template('home.html', title="Home")

@app.route("/results")
def results():
    # example data
    # data = [{
    #     'title': ['Presentation 1', 'Presentation 2', 'Presentation 3'],
    #     'page_url': ['https://example.com/1', 'https://example.com/2', 'https://example.com/3'],
    #     'likes': [100, 150, 80],
    #     'recorded_date': ['2024-01-01', '2024-02-15', '2024-03-10'],
    #     'speakers': ['Speaker A', 'Speaker B', 'Speaker C'],
    #     'topics': ['Topic X', 'Topic Y', 'Topic Z'],
    #     'views': [1000, 1200, 800],
    #     'summary': ['Summary of Presentation 1', 'Summary of Presentation 2', 'Summary of Presentation 3']
    # }]
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
    # for result in data.iloc:
    #     print("START")
    #     print(result)
    # print(data.iloc[0].title)
    # print(data.head())
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