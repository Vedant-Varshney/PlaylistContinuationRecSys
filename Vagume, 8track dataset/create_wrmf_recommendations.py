import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import implicit
from tqdm.notebook import tqdm, trange

from sklearn.model_selection import train_test_split

import importlib

from collections import defaultdict
from importlib import reload

import pickle

from joblib import delayed, Parallel

import seaborn as sns
sns.set()

import sys
sys.path.insert(0,"..")

import df_utils
import mf_utils

def main():
    df = pd.read_csv("cleaned_data_0.2.csv", index_col=[0])
    songs_df = pd.read_csv("songs_DF.csv", index_col=[0])

    split_train_df = pd.read_csv("Split_Train_DF.csv", index_col=[0])
    excl = pickle.load(open("Excluded_Songs_Series", "rb"))

    split_train_ratings_matrix = mf_utils.df_to_sparse(split_train_df, all_songs=df.spotify_id.unique())

    # # Number of latent factors now 300
    # model = implicit.als.AlternatingLeastSquares(factors=300)
    #
    # # items_users matrix in this case is the songs_playlist matrix
    # model.fit(split_train_ratings_matrix)
    #
    # pickle.dump(model, open("WRMF_model_factors300.p", "wb"))

    model = pickle.load(open("WRMF_model_factors300.p", "rb"))

    # CANNOT use songs_df here as order is different
    indx_to_song = mf_utils.indx_mapping(df.spotify_id.unique())
    indx_to_playlist = mf_utils.indx_mapping(split_train_df.playlist_id.unique())

    song_to_indx = mf_utils.indx_mapping(df.spotify_id.unique(), indx_to_item=False)
    playlist_to_indx = mf_utils.indx_mapping(split_train_df.playlist_id.unique(), indx_to_item=False)

    wrmf_recs_df, wrmf_score_df = mf_utils.gen_wrmf_candidates(model, split_train_ratings_matrix,
    split_train_df.playlist_id.unique(), playlist_to_indx, indx_to_song)

    wrmf_recs_df.to_csv("WRMF_Candidate_Spotify_IDs.csv")
    wrmf_score_df.to_csv("WRMF_Candidate_Scores.csv")

if __name__ == '__main__':
    main()
