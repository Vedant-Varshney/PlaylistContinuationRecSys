#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:43:00 2020

@author: vedantvarshney
"""

import requests
import json
from collections import namedtuple
import pandas as pd
from copy import copy
from tqdm import trange

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

#
#results = sp.search(q='Energy (Feat. Juicy J)', limit=2)
#print(results)
#spot_ids = []
#for idx, track in enumerate(results['tracks']['items']):
#    print(idx, track['name'])
#    print(track["artists"][0]["name"])
#    print(track["album"]['name'])

#    spot_ids.append(track["id"])
#
#
#audio_features = sp.audio_features(spot_ids)
#print(audio_features)



def json_object_hook(Dict):
    return namedtuple("X", Dict.keys())(*Dict.values())

def objectify(resp):
    return json.loads(resp,
                      object_hook=json_object_hook
                      )


def batch(iterable, size=1):
    length = len(iterable)

    for indx in range(0, length, size):
        yield iterable[indx: min(indx + size, length)]


def preproc(fpath, append=False):
    chunksize = 1000
    chunk_num = 0
    skiprows = 665993

    for chunk in pd.read_csv(fpath,
                              chunksize = chunksize,
                              iterator = True,
                              skiprows = range(1,skiprows),
                              dtype = {
                                      "spotify_id": str,
                                      "album": str
                                      }
                              ):

        chunk_num += 1
        print(f"CHUNK: {chunk_num}")

        chunk.dropna(axis=1, how="all", inplace=True)
        chunk.dropna(inplace=True)

        for col in ["a", "mbids"]:#, "Unnamed: 0", "Unnamed: 9"]:
            chunk.drop(col, axis=1, inplace=True)

        chunk_length = len(chunk)
        print(f"chunk_length : {chunk_length}")

        batch_size = 100

        batch_num = 0

        for i in trange(0, chunk_length, batch_size):

            batch_num += 1
#            empty_search_flag = False

            print(f"Chunk: {chunk_num}")

            names = chunk['track_name'][i: min(i + batch_size, chunk_length)]
            artists = chunk['artist_name'][i: min(i + batch_size, chunk_length)]

            length = len(names)
            filt_length = copy(length)

            metrics = ["danceability", "energy", "loudness", "speechiness",
                       "acousticness", "instrumentalness","liveness", "valence",
                       "tempo"]

            for j, name in enumerate(names):
                try:
                    artist = artists[j]
                    query = f"{name} - {artist}"
                except:
                    query = copy(name)
#
                tracks = sp.search(q=query, limit=1)['tracks']['items']

                if len(tracks) == 0:
                    print("empty search hit")
                    filt_length -= 1
#                    empty_search_flag = True
                    continue

                else:
                    track = tracks[0]

                    chunk["spotify_id"].iat[i+j] = track["id"]
                    chunk["album"].iat[i+j] = track["album"]['name']
                    chunk["popularity"].iat[i+j] = track["popularity"]


#            if empty_search_flag:
#                continue

            #retrieve audio features of 100 tracks in one go

            spot_ids = chunk["spotify_id"][i:i+length].tolist()

            filt_ids = [v for v in spot_ids if pd.notna(v)]

            audio_features = sp.audio_features(filt_ids)


            #TODO - inefficient?
            for af in audio_features:
                if af != None:
                    indx = spot_ids.index(af["id"])
                    for metric in metrics:
                        chunk[metric].iat[i+indx] = af[metric]

        if chunk_num == 1 and append == False:
            header = True
        else:
            header = False

        chunk.to_csv("cleaned_data.csv", mode='a', header=header)


def clean(fpath):
    chunksize = 10000
    i=0

    for chunk in pd.read_csv(fpath,
                             chunksize = chunksize,
                             iterator = True,
#                             skiprows=544996,
                             dtype = {
                                      "spotify_id": str,
                                      "album": str
                                      }
                              ):

        for col in ["a", "mbids", "Unnamed: 0", "Unnamed: 9"]:
            chunk.drop(col, axis=1, inplace=True)

        chunk.dropna(axis=1, how="all", inplace=True)
        chunk.dropna(inplace=True)

        if i == 0:
            header = True
        else:
            header = False

        chunk.to_csv("cleaned_data.csv", mode="a", header=header)

        i+=1

        print(f"CHUNK SAVED: {i}")


if __name__ == "__main__":
    smallpath = "small_data.csv"
    bigpath = "MPSD v1.0.csv"
    my_data_path = "data.csv"

    preproc(bigpath, append=True)
#    clean(my_data_path)










    
