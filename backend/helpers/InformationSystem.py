import numpy as np
import pandas as pd
import os
import json
from googleapiclient.discovery import build
from dotenv import load_dotenv
import BuildMatrix

# Load .env files
load_dotenv()
API_KEY = MY_ENV_VAR = os.getenv('API_KEY')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the TED Talk transcript data as a Pandas dataframe
    """
    with open('backend/init.json', 'r') as json_file:
        data = json.load(json_file)
        df = pd.json_normalize(data)
    docname_to_idx, word_occ_mat = BuildMatrix.df_to_word_occ_mat(df)
    return docname_to_idx, word_occ_mat


def submit_query(query: str) -> pd.DataFrame:
    """
    Submit a query to the IR System and return a dataframe consisting of every
    video ranked in order of relevancy
    """
    pass


def related_videos(id: str) -> pd.DataFrame:
    """
    Given a video id, returns a dataframe consisting of every video ranked in
    order of relevancy
    """
    pass


def get_video_comments(video_id, api_key=API_KEY) -> list[dict]:
    """
    Takes a YouTube Video ID and returns the top 20 comments in a list of the form:
    {
        username: str,
        comment_likes: int,
        body: str
    }
    If comments are disabled returns 'ERROR'
    """
    comments = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.commentThreads().list(
        part='snippet',
        order='relevance',
        videoId=video_id
    )
    count = 0
    try:
        response = request.execute()
        while response and count < 20:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                body = comment['textDisplay']
                author = comment['authorDisplayName']
                likes = comment['likeCount']
                comments.append({
                    'username': author,
                    'comment_likes': likes,
                    'body': body
                })
                count += 1
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part='snippet',
                    order='relevance',
                    videoId=video_id,
                    pageToken=response['nextPageToken']
                )
                response = request.execute()
            else:
                break
        return comments
    except:
        return "ERROR"


def convert_likes_to_int(likes) -> int:
    """
    Converts likes (of the form {number}{Letter}) to integer by 
    - multiplying by 1000 if letter is K
    - multiplying by 1000000 if letter is M
    - leaving likes as is otherwise
    """
    if 'K' in likes:
        num = int(float(likes.replace('K', '')) * 1000)
    elif 'M' in likes:
        num = int(float(likes.replace('M', '')) * 1000000)
    else:
        num = int(likes)
    return num


def preprocess_data() -> None:
    """
    Converts data in talks_info.csv into talks_info_final.csv with the following
    changes:

    - likes are converted into integers
    - new column for "reviews", consisting of YouTube comments about the talk where
      each review has the following format for each object:
      {
        username: str,
        comment_likes: int,
        body: str
      }
    """
    ted_talks = pd.read_csv("static/data/talks_info.csv")
    ted_talks['comments'] = ted_talks['youtube_video_code'].apply(
        func=get_video_comments)
    ted_talks['likes_int'] = ted_talks['likes'].apply(convert_likes_to_int)
    ted_talks.to_csv("talks_info_final.csv")


def sentiment_analysis(review: str) -> bool:
    """
    Given a review, returns True if the review has a positive sentiment and
    False if the review has a negative sentiment
    """
    pass
