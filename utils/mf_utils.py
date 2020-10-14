"""
Module containing utility functions for calculating various metrics and
performing common operations relevant to matrix factorisation.

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

def indx_mapping(arr, indx_to_item=True):
    """
    Maps a unique array to a dictionary.

    If indx_to_item:
        keys = array index, value = array item.
    Else other way round.
    """
    if indx_to_item:
        return {i: x  for (i, x) in enumerate(arr)}
    else:
        return {x: i for (i, x) in enumerate(arr)}

def recs_to_spotifyids(recommendations, indx_to_song):
    """
    Translates recommendations to a dict of Spotify IDs and scores.

    Arguments:
        - recommendations - as returned by implicit
        - indx_to_song - dict mapping song index to Spotify ID

    Returns:
        - dict where key = Spotify ID, value = score
    """

    return {indx_to_song[id_]: score for (id_, score) in recommendations}

def inspect_recommendations(playlist_indx, recommendations, df, songs_df,
                            indx_to_playlist=None, indx_to_song=None):
    """
    Inspect a given set of recommendation by printing the relevant songs and
    their metadata.

    Arguments:
        - playlist_indx - index of target playlist in ratings matrix
        - recommendations - recommendations as returned by implicit
        - df - original dataframe (cleaned_data_0.2.csv)
        - songs_df - spotify ID indexed dataframe for all songs and corresp. metadata
        (songs_DF.csv)
        - indx_to_playlist - dictionary mapping index of a playlist in the ratings
        matrix to the playlist_id
    """

    # Create indx_to_playlist hashmap if not provided
    if indx_to_playlist is None:
        indx_to_playlist = indx_mapping(df.playlist_id.unique())

    if indx_to_song is None:
        indx_to_song = indx_mapping(songs_df.index.unique())

    sample_playlist = indx_to_playlist[playlist_indx]

    liked_song_ids = df[df.playlist_id == sample_playlist].spotify_id.unique()

    print("Liked Tracks:")
    print(songs_df.loc[liked_song_ids][["track_name", "artist_name", "popularity"]])

    recs_ids_dict = recs_to_spotifyids(recommendations, indx_to_song)

    recs = songs_df.loc[list(recs_ids_dict.keys())][["track_name", "artist_name", "popularity"]]
    recs["score"] = recs_ids_dict.values()

    print("\nRecommended Tracks:")
    print(recs)


def df_to_sparse(df, all_songs):
    """
    Converts a dataframe to a sparse implicit ratings matrix.

    Arguments:
        - df - dataframe to convert to a sparse matrix
        - all_songs - list of all songs in the search space
        - withold_dict - a dictionary whose keys are playlist IDs and whose values are a list of songs which should be
        excluded from that playlist when calculating the implicit ratings matrix.

    Returns:
        - ratings_matrix - implicit ratings matrix in CSR format. Row indx = song indx; column indx = playlist indx
    """
    # Create hashmap of the list of unique songs for fast index lookup
    # np.where etc will unnecessarily search the entire array and thus will not scale well.
    song_to_indx = indx_mapping(all_songs, indx_to_item=False)

    # Mapping each song in the original DF to the index in the unique songs list
    song_indxs = df.spotify_id.apply(lambda id_: song_to_indx[id_]).values

    # Same for list of unique playlists
    playlist_to_indx = indx_mapping(df.playlist_id.unique(), indx_to_item=False)

    # Mapping each playlist in the original DF to the index in the unique playlists list
    playlist_indxs = df.playlist_id.apply(lambda id_: playlist_to_indx[id_]).values

    data = np.ones(df.shape[0])

    assert data.shape == song_indxs.shape == playlist_indxs.shape

    # Although matrix only contains int, cast as float for safety in future calculations
    # row indx = song indx
    # column indx = playlist indx
    ratings_matrix = sp.sparse.csr_matrix((data, (song_indxs, playlist_indxs)), dtype=np.float64)

    return ratings_matrix


def hit_rate(recommendations, excluded_songs, indx_to_song):
    """
    Returns the hit rate (correct recommendations/number of songs excluded)

    Arguments:
        - recommendations - recs for a single playlist as returned by implicit
        - excluded_songs - Spotify IDs excluded from that playlist
    """
    recs_spotifyids = list(recs_to_spotifyids(recommendations, indx_to_song).keys())
    num_hits = len(set(recs_spotifyids).intersection(set(excluded_songs)))

    return num_hits/len(excluded_songs)


def calc_hit_rates(model, item_user_matrix, excl, playlist_to_indx, indx_to_song,
                   N=20000, parallelise=False, progressbar=True):
    """
    Calculates hit rate for all playlists in excl.

    Arguments:
        - model - implicit model
        - item_user_matrix - sparse CSR matrix for model
        - excl - dict. of the playlists and songs excluded in matrix. key = playlist; value = list of songs
        - playlist_to_indx - dict. mapping playlist ID to index in playlists array as fed to implicit model
        - parallelise - distribute job over 16 worker threads
        - progressbar - show progress bar

    Returns:
        - list of tuples. [ (playlist_id, hit_rate_for_that_playlist), ... ]
    """

    if progressbar:
        iter_excl = tqdm(excl.items(), total=len(excl))
    else:
        iter_excl = excl.items()

#     Nested helper to parallelise
    def helper(excl_playlist, excl_songs):
        recs = model.recommend(playlist_to_indx[excl_playlist], item_user_matrix.T, N=N)
        return (excl_playlist, hit_rate(recs, excl_songs, indx_to_song))

    if parallelise:
        hit_rates = \
        Parallel(n_jobs=16, prefer="threads")(delayed(helper)(id_, songs) for (id_, songs) in iter_excl)
    else:
        hit_rates = []
        for (id_, songs) in iter_excl:
            hit_rates.append(helper(id_, songs))

    return hit_rates



def grid_search_factors(srange, item_user_matrix, excl, playlist_to_indx,
                       parallelise=True):

    def helper(factors):
        model = implicit.als.AlternatingLeastSquares(factors=factors)

        # items_users matrix in this case is the songs_playlist matrix
        model.fit(item_user_matrix, show_progress=False)

        temp_hit_rates = calc_hit_rates(model, item_user_matrix, excl, playlist_to_indx,
                                        parallelise=False, progressbar=False)

        avg_hit_rate = np.mean([hr for _, hr in temp_hit_rates])

        print(f"factors {factors} | hit rate {avg_hit_rate}")

        return (factors, avg_hit_rate)

    iter_factors = tqdm(srange, total=len(srange))

    if parallelise:
        hit_rates = Parallel(n_jobs=4, prefer="threads")(delayed(helper)(factors) for factors in iter_factors)
    else:
        hit_rates = [helper(factors) for factors in iter_factors]

    return hit_rates



def tuples_to_dict(list_tuples):
    """
    Converts a list of tuples [(first, second), (first, second), ...]
    to a dictionary {first: second, first: second, ...}

    Arguments:
        - list_tuples
    """
    return {k: v for (k,v) in list_tuples}


def unpack_tuples_list(list_tuples):
    """
    Converts a list of tuples [(first, second), (first, second), ... ]
    into separate lists [[first, first, ...] , [second, second, ...]]
    """
    temp_dict = defaultdict(list)

    for tup in list_tuples:
        for i, elem in enumerate(tup):
            temp_dict[i].append(elem)

    return temp_dict.values()


def gen_wrmf_candidates(model, item_user_matrix, playlist_ids, playlist_to_indx, indx_to_song, N=20000):
    """
    Generates N recommendations for each playlist.

    Arguments:
        - model - pre-trained implicit model
        - item_user_matrix - sparse CSR item-user matrix
        - playlist_ids - list of unique playlist IDs to generate recommendations for
        - playlist_to_indx - dict. where key = playlist ID, value = index in matrix
        - indx_to_song - dict. where key = index in matrix, value = song Spotify ID
        - N - number of recommendations to generate per playlist

    Returns:
        - wrmf_recs_df - pandas DF containing recommended Spotify IDs. columns = playlist IDs
        - wrmf_score_df - pandas DF containing scores for above dataframe.
    """
    def helper(playlist_id):
        playlist_indx = playlist_to_indx[playlist_id]

#         Need to provide user-item matrix here
        recs = model.recommend(playlist_indx, item_user_matrix.T, N=N)
        rec_indxs, rec_scores = unpack_tuples_list(recs)

        rec_spot_ids = [indx_to_song[x] for x in rec_indxs]

        return (playlist_id, rec_spot_ids, rec_scores)


    play_ids, spot_ids, scores = unpack_tuples_list(
        Parallel(n_jobs=4, prefer="threads")(delayed(helper)(id_) for id_ in tqdm(playlist_ids)))

#     Quick check to ensure order was maintained in the above parallelised job
    assert (pd.Series(play_ids) == pd.Series(playlist_ids)).all()

    wrmf_recs_df = pd.DataFrame(columns=playlist_ids, data=np.asarray(spot_ids).T)
    wrmf_score_df = pd.DataFrame(columns=playlist_ids, data=np.asarray(scores).T)

    return wrmf_recs_df, wrmf_score_df
