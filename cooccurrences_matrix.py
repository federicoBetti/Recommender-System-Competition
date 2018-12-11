import gc
import itertools
import os
import pickle

from scipy.sparse import csr_matrix
import scipy.sparse as sparse

from Dataset.RS_Data_Loader import RS_Data_Loader
import pandas as pd
import numpy as np


if __name__ == '__main__':



    target_playlist = pd.read_csv(os.path.join("Dataset", "target_playlists.csv"))
    train = pd.read_csv(os.path.join("Dataset", "train.csv"))
    tracks = pd.read_csv(os.path.join("Dataset", "tracks.csv"))

    target_playlist_m = np.array(target_playlist.playlist_id.as_matrix())
    songs_per_playlist = pd.Series(train.groupby(train.playlist_id).track_id.apply(list))
    songs_number = train.groupby(train.track_id).track_id.groups

    relation_matrix = np.zeros([len(tracks), len(tracks)], dtype=int)

    all_combinations = []
    for pl in songs_per_playlist:
        # sort it to have the pairs in order.
        # In this way you'll build a traingular matrix and and the end you can easilly sum it with its transpose
        # to create the final one (in which diagonal values will not be present yet)
        pl.sort()
        all_combinations.append(list(itertools.combinations(pl, 2)))

    all_combinations = [item for sublist in all_combinations for item in sublist]

    i = 0
    # simply, add 1 in all pairs that have been found
    for el in all_combinations:
        i += 1
        relation_matrix[el] += 1

    relation_matrix_complete = relation_matrix + relation_matrix.transpose()
    del relation_matrix
    print(np.nonzero(relation_matrix_complete))

    i = 0
    for line in relation_matrix_complete:
        # fill the diagonal
        relation_matrix_complete[i, i] = len(songs_number[i])
        i += 1
    del tracks
    del train
    del target_playlist
    del target_playlist_m
    del songs_number
    del songs_per_playlist
    gc.collect()
    cooc_csr = csr_matrix(relation_matrix_complete)
    sparse.save_npz(os.path.join("Dataset", "cooc_matrix.npz"), cooc_csr)
