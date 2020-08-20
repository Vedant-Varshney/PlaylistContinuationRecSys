"""
Module containing utility functions for calculating various metrics and
performing common operations relevant to matrix factorisation.

Note - module currently assumes the standard df column names.
Consider changing this later.
"""


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
        indx_to_playlist = {i:playlist  for (i, playlist) in enumerate(df.playlist_id.unique())}

    if indx_to_song is None:
        indx_to_song = {i:song  for (i, song) in enumerate(songs_df.index.unique())}

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


def df_to_sparse(df, all_songs):
    """
    Converts a dataframe to a sparse implicit ratings matrix.

    Arguments:
        - df - dataframe to convert to a sparse matrix
        - all_songs - list of all songs in the search space
        - withold_dict - a dictionary whose keys are playlist IDs and whose values are a list of songs which should be
        excluded from that playlist when calculating the implicit ratings matrix.

    Returns:
        - ratings_matrix - implicit ratings matrix in CSR format
    """
    # Create hashmap of the list of unique songs for fast index lookup
    # np.where etc will unnecessarily search the entire array and thus will not scale well.
    song_to_indx = {song: i for (i, song) in enumerate(all_songs)}

    # Mapping each song in the original DF to the index in the unique songs list
    song_indxs = df.spotify_id.apply(lambda id_: song_to_indx[id_]).values

    # Same for list of unique playlists
    playlist_to_indx = {playlist: i for (i, playlist) in enumerate(df.playlist_id.unique())}

    # Mapping each playlist in the original DF to the index in the unique playlists list
    playlist_indxs = df.playlist_id.apply(lambda id_: playlist_to_indx[id_]).values

    data = np.ones(df.shape[0])

    assert data.shape == song_indxs.shape == playlist_indxs.shape

    # Although matrix only contains int, cast as float for safety in future calculations
    # row indx = song indx
    # column indx = playlist indx
    ratings_matrix = sp.sparse.csr_matrix((data, (song_indxs, playlist_indxs)), dtype=np.float64)

    return ratings_matrixd
