"""
Module containing utility functions for calculating various metrics and
performing common operations relevant to the RNN models in this project.

Note - module currently assumes the standard df column names.
Consider changing this later.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import implicit
from tqdm.notebook import tqdm, trange

from sklearn.model_selection import train_test_split

from collections import defaultdict
from importlib import reload

import pickle

from joblib import delayed, Parallel


def create_playlist_song_matrix(df, song_to_indx,
                                song_vocab=None, playlists=None):
    """
    Creates a matrix consisting of playlist arrays which in turn contain
    the *latent vectors* of the songs within that playlist.

    IMPORTANT NOTE - only testable playlists (length > 1) are included in matrix.

    Also returns a length dictionary wrapped in a Pandas series.

    Args:
        - df - standard 'cleaned_data' DataFrame
        - song_to_indx - dict. mapping Spotify ID to its index
        - song_vocab (optional) - list of Spotify IDs which defines the vocab.
        for a model. If None, vocab treated as unlimited.
        - playlists - playlists to process.
        If None, playlists = df.playlist_id.unique()

    Returns:
        - playlist_song matrix
        - length_dict - Pandas series where key = all playlists processed,
        value = length of that playlist after vocab mask is applied.
    """

    if playlists is None:
        playlists = df.playlist_id.unique()

    length_dict = {}
    matrix = []
    for id_ in tqdm(playlists):
        temp_songs = df[df.playlist_id == id_].spotify_id

        if song_vocab is not None:
            vocab_mask = temp_songs.isin(song_vocab)
            temp_songs = temp_songs[vocab_mask]

        temp_songs = temp_songs.apply(lambda x: song_to_indx[x]).values

        length_dict[id_] = len(temp_songs)

#         All testable playlists have a length greater than 1
        if len(temp_songs) > 1:
            matrix.append(temp_songs)

    return np.asanyarray(matrix), pd.Series(length_dict)


def sample_split(arr):
    """
    Randomly splits an array into two. The only requirement is that
    neither of the two arrays should be empty.

    Args:
        - array-like (supports len, slicing)

    Returns:
        - (arr1, arr2)
    """
    length = len(arr)

    assert length > 1

    if length == 2:
        sample_indx = 1
    else:
        sample_indx = np.random.choice(range(1, len(arr)-1))

    return (arr[:sample_indx], arr[sample_indx:])

def prepare_sample(playlist, latent_songs):
    x, y = sample_split(playlist)

    x = pd.Series(x).apply(lambda indx: latent_songs[indx])
    x = pd.DataFrame.from_dict(dict(zip(x.index, x.values))).T.to_numpy()

    return (x, y)
