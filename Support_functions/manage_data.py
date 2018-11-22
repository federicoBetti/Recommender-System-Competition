import pandas as pd
import gc
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import numpy as np
from itertools import repeat
import itertools
from scipy import sparse
from scipy.sparse import csr_matrix
import os
from threading import Lock


def create_relation_matrix(train, tracks):
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

    # simply, add 1 in all pairs that have been found
    for el in all_combinations:
        relation_matrix[el] += 1

    relation_matrix_complete = np.add(relation_matrix, relation_matrix.transpose())
    for i in range(relation_matrix_complete.shape[0]):
        try:
            relation_matrix_complete[i, i] = len(songs_number[i])
        except KeyError:
            relation_matrix_complete[i, i] = 0

    return csr_matrix(relation_matrix_complete)


def get_icm_matrix(tracks):
    tracks_arr = tracks.track_id.values
    album_arr = tracks.album_id.unique()
    artists_arr = tracks.artist_id.unique()
    feature_tracks = np.ndarray(shape=(tracks_arr.shape[0], album_arr.shape[0] + artists_arr.shape[0]))

    def create_feature(entry):
        feature_tracks[entry.track_id][entry.album_id] = 1
        feature_tracks[entry.track_id][album_arr.shape[0] + entry.artist_id] = 0.8

    tracks.apply(create_feature, axis=1)
    return csr_matrix(feature_tracks)


def get_data():
    train = pd.read_csv("Dataset\\train.csv")
    tracks = pd.read_csv("Dataset\\tracks.csv")
    target_playlist = pd.read_csv("Dataset\\target_playlists.csv")
    return train, tracks, target_playlist


def formatting_reccomendation(result_list):
    return " ".join(str(x) for x in result_list)


# target playlist with 'tracks_ids' column as a list of element to add
def make_CSV_file(target_playlist, filename):
    target_playlist.set_index('playlist_id', inplace=True)
    target_playlist['track_ids'] = target_playlist['track_ids'].apply(formatting_reccomendation)
    target_playlist.to_csv(os.path.join('Results', filename))


def get_submission_file(filename):
    submission_exp = pd.read_csv(os.path.join('Results', filename))
    return submission_exp


def garbage_collect():
    gc.collect()


def rating_computation(function_to_run, similarity_matrix, train, target_playlist, thread_number=1, URM=None):
    dict_train = train.groupby(train.playlist_id).track_id
    songs_per_target_playlist = pd.Series(dict_train.apply(list))[target_playlist.playlist_id]

    if thread_number == 1:
        tmp = songs_per_target_playlist.apply(
            partial(function_to_run, URM=URM, similarity_matrix=similarity_matrix))
        target_playlist['track_ids'] = list(tmp)
    else:
        songs_per_target_playlist = list(songs_per_target_playlist)
        pool = ThreadPool(thread_number)
        results = pool.starmap(function_to_run, zip(songs_per_target_playlist, repeat(URM), repeat(similarity_matrix)))
        target_playlist['track_ids'] = results
    return target_playlist


def compute_cosine_similarity(matrix, shrink=0):
    similarity = (matrix.T).dot(matrix)
    a = np.sqrt(similarity.diagonal())
    den = similarity.copy().transpose()
    den.data = np.ones(den.data.shape)
    # den = (den * a).T * a
    den = (den.multiply(a)).transpose().multiply(a)
    # per fare questo non puo usare direttamente le matrici sparse come prima quindi la memoria tende a esplodere

    if shrink != 0:
        den.data += shrink
        print("shrik applied")

    den = den.tocsr()
    similarity.data = np.divide(similarity.data, den.data)
    print("division done")
    del a
    del den
    gc.collect()
    return similarity


lock = Lock()
def write_txt(filename, txt):
    with lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write("%s\n" % txt)


def get_fake_test():
    return sparse.load_npz(os.path.join("Intermediate_values", "fake_test.npz"))


def assign_recomendations_to_correct_playlist(target_playlist, target_recommendations):
    target_playlist['track_ids'] = pd.Series(['0'] * len(target_playlist.index), index=target_playlist.index)
    for user_id, recommendations in target_recommendations:
        ind = target_playlist[target_playlist.playlist_id == user_id].index[0]
        target_playlist.at[ind, 'track_ids'] = recommendations