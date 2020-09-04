import requests
import pandas as pd
from copy import copy
from tqdm import trange, tqdm
import numpy as np

import spotipy
from spotipy.oauth2 import SpotifyOAuth


client_id = "16dab5df57af40a0b913a3010f3b34ba"
client_secret = "65223edc47de4e6b84c794d901f1d011"

scope = "user-library-read"
redirect_uri = "http://collab_vedant.com/callback/"
username =  "jp23jhto8yfwq8rsv6yyg4wg8"

auth = SpotifyOAuth(client_id=client_id,
                           client_secret=client_secret,
                           scope=scope,
                           redirect_uri=redirect_uri,
                           username=username)


sp = spotipy.Spotify(auth_manager=auth, requests_timeout=100)

class NoTracks(Exception):
    """
    Exception - no track results for query.
    """
    pass

def get_top_track(query):
    tracks = sp.search(q=query, limit=1)['tracks']['items']
    if len(tracks) == 0:
        raise NoTracks
    else:
        return tracks[0]

def process(row, new_columns=None, audio_metrics=None):
    new_data = pd.Series(index=new_columns)

    query = f"{row.track_name} - {row.artist_name}"

    try:
        track = get_top_track(query)
    except NoTracks:
        return new_data

    audio_features = sp.audio_features(track["id"])[0]

    new_data["spotify_id"] = track["id"]
    new_data["album"] = track["album"]["name"]
    new_data["popularity"] = track["popularity"]

    for metric in audio_metrics:
        try:
            new_data[metric] = audio_features[metric]
        except:
            new_data[metric] = np.nan

    return new_data


def create_new_df(src_df, dest_file):
    audio_metrics = ["danceability", "energy", "loudness", "speechiness",\
    "acousticness", "instrumentalness","liveness", "valence", "tempo"]

    new_columns = ["spotify_id", "album", "popularity"] + audio_metrics

    print("Source loaded")

    tqdm.pandas()

    src_df = src_df.join(
    src_df.progress_apply(process, new_columns=new_columns, audio_metrics=audio_metrics, axis=1)
    )

    src_df.to_csv(dest_file)


class InvalidMask(Exception):
    pass

def append_df(src_df, append_file, dest_file):
    last_row = pd.read_csv(append_file).tail(1).iloc[0]
    print("File to append loaded")

    mask = (src_df.track_name == last_row.track_name) & \
    (src_df.playlist_id == last_row.playlist_id)

    masked_src = src_df[mask]

    if len(masked_src) != 1:
        raise InvalidMask

    create_new_df(src_df[masked_src.index[0]:], dest_file)


def test_create_new_df():
    create_new_df(src_df, "preprocessed_data_full.csv")

def main():
    src_df = pd.read_csv("original_data.csv")
    print("Source loaded")

    append_df(src_df, "cleaned_data_0.2.csv", "appended_data.csv")

if __name__ == '__main__':
    main()
    # main("original_data.csv", "preprocessed_data_full.csv")
    # test_main()
