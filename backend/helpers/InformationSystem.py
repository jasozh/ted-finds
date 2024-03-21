import numpy as np
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the TED Talk transcript data as a Pandas dataframe
    """
    return pd.read_csv(file_path)

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
    pass

def sentiment_analysis(review: str) -> bool:
    """
    Given a review, returns True if the review has a positive sentiment and
    False if the review has a negative sentiment
    """
    pass