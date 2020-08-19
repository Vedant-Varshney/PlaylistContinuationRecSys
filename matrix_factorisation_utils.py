"""
Module containing utility functions for calculating various metrics and
performing common operations relevant to matrix factorisation.

Note - module currently assumes the standard df column names.
Consider changing this later.
"""


def inspect_recommendations(playlist_indx, recommendations, df=df, songs_df = songs_df,
                            indx_to_playlists=indx_to_playlists):
    """
    Inspect a given set of recommendation by printing the relevant songs and
    their metadata.

    Arguments:
        - playlist_indx - index of target playlist in ratings matrix
        - recommendations - recommendations as returned by implicit
        - df - original dataframe (cleaned_data_0.2.csv)
        - songs_df - spotify ID indexed dataframe for all songs and corresp. metadata
        (songs_DF.csv)
        - indx_to_playlists - dictionary mapping index of a playlist in the ratings
        matrix to the playlist_id
    """
    sample_playlist = indx_to_playlist[playlist_indx]

    liked_song_ids = df[df.playlist_id == sample_playlist].spotify_id.unique()

    print("Liked Tracks:")
    print(songs_df.loc[liked_song_ids][["track_name", "artist_name", "popularity"]])

    rec_song_ids = []
    rec_scores = []

    for id_, score in recommendations:
        rec_song_ids.append(indx_to_song[id_])
        rec_scores.append(score)

    recs = songs_df.loc[rec_song_ids][["track_name", "artist_name", "popularity"]]
    recs["score"] = rec_scores

    print("\nRecommended Tracks:")
    print(recs)
