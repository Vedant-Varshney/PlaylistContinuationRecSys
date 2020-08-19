# Problem Statement and Strategy

Notation:

caps - fixed size space/vector

lowercase - variable size space/vector

## Problem Definition:

We have a set of _M_ playlists each consisting of a variable number of _k_ songs.
In total, there are _S_ unique songs in the dataset.

Each song has a feature vector _F_. This vector contains numerical features
(song characteristics) and categorical features (playlist name, artist name,
album name etc.) of a high cardinality.

The task is to recommend new songs which a user is likely to enjoy given an input
playlist.

In the ideal case, not only should the recommended songs be appropriate given the
playlist as a whole but they should also follow the _flow_ of the
playlist; i.e. they should continue on any trends down the playlist.

## Dataset:

For the subset of MSPD dataset considered here (cleaned_data.csv):

_M_ - contains ~10,000 playlists

_k_ - contains a long tail from ~2800 to 0 songs (see Data_Exploration)

_S_ - ~100,000 unique songs

_F_
  - song_characteristics - array(float), length=10
  - source - categorical, cardinality=2
  - user_id - categorical, cardinality~4,000
  - tags - array(categorical), cardinality~2,000
  - artist_name - categorical, cardinality~12,000
  - album - categorical, cardinality~55,000


## SWOT Analysis:

*Strengths:*
- The current dataset is sufficiently large such that it may be possible to
make use of ML methods
- A wide range of users (approx. 10,000) is covered. The resulting model therefore
may be quite general.

*Weaknesses:*
- Very prominent cold start problem (many playlists in the dataset are very short).
Approx. 45% of playlists contain less than 15 songs and 10% contain less than 5.
Such a length may not be sufficiently long so as to make clear any important trends.
- The dataset was sourced from Vagume & 8track, music streaming providers known
for catering to a more niche audience. This selection bias may mean that result in
our model being less generalisable to the masses.
- We may choose to recommend songs from not just the songs in the training data
but rather the entire Spotify library. This could severely impact model scalability.
- The high dimensionality of the features of each song may prevent our model from
learning to recognise the important information.

*Opportunities:*
- The dataset can be readily expanded should we find that the amount of data available
is limiting model performance. The data can be expanded to include all of MPDS
and other sources (e.g. the NowPlaying dataset). This will, however, likely be very
time consuming as the features will need to be built using Spotify's Web API which
is request rate limited.

*Threats:*


## Strategy

Note - Approach inspired by that taken by the winners of the 2018 ACM RecSys
Challenge. See paper 'Two-stage Model for Automatic Playlist Continuation at Scale'.

*Step 1.* Search space reduction and songs embeddings.
- Reduce the total number of searchable songs from ~100,000 (in current dataset) to
approx. ~20,000 _for each playlist_ with as high a recall as possible.
- Consider including more recent songs from the wider Spotify library before the space reduction.
- Do not use the song characteristics here, we will use that later.

- Construct an implicit feedback, playlist-song matrix. 1 if song i in playlist j. 0 otherwise.
- Build a WRMF model to factorise the matrix and find the latent embeddings.
- Find the top 20,000 songs for each playlist and use these from now on.


*Step 2.* Playlist embeddings.
- Use a conditional LSTM/GRU model to build a new set of playlist embeddings from the song
embeddings (only 20,000 per playlist)
- Use a learned representation of the categorical data (artist_name, album_name, tags?)
to set an/the initial state for the first LSTM cell.
- Output an embedding for each playlist. The goal here is to capture the information
contained in the overall flow of the playlist, not just its constituent songs.
- Proceed with these new playlist embeddings or blend with those from WRMF.

*Step 3.* Gradient boosted trees for final song ranking per playlist.
- Can finally use song characteristics as features.
-
