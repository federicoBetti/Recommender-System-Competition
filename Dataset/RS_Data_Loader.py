#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import os
import random

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sps
import time
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle

from sklearn.feature_extraction.text import TfidfTransformer

from Support_functions import get_evaluate_data as ged


def create_URM_matrix(train):
    row = list(train.playlist_id)
    col = list(train.track_id)
    return csr_matrix(([1] * len(row), (row, col)), shape=(50446, 20635))


def get_icm_matrix(tracks):
    tracks_arr = tracks.track_id.values
    album_arr = tracks.album_id.unique()
    artists_arr = tracks.artist_id.unique()
    feature_tracks = np.ndarray(shape=(tracks_arr.shape[0], album_arr.shape[0] + artists_arr.shape[0] + 1))

    def encode_duration_sec(duration):
        if duration < 120:
            return 0
        elif 120 < duration < 320:
            return 1
        elif 320 < duration < 420:
            return 2
        else:
            return 3

    def create_feature(entry):
        feature_tracks[entry.track_id][entry.album_id] = 1
        feature_tracks[entry.track_id][album_arr.shape[0] + entry.artist_id] = 0.8
        duration_encoded = encode_duration_sec(entry.duration_sec)
        feature_tracks[entry.track_id][
            album_arr.shape[0] + artists_arr.shape[0]] = duration_encoded

    tracks.apply(create_feature, axis=1)
    to_ret = csr_matrix(feature_tracks)
    del feature_tracks
    return to_ret


def compute_PageRank(G, beta=0.85, epsilon=10 ** -4):
    '''
    Efficient computation of the PageRank values using a sparse adjacency
    matrix and the iterative power method.

    Parameters
    ----------
    G : boolean adjacency matrix. np.bool8
        If the element j,i is True, means that there is a link from i to j.
    beta: 1-teleportation probability.
    epsilon: stop condition. Minimum allowed amount of change in the PageRanks
        between iterations.

    Returns
    -------
    output : tuple
        PageRank array normalized top one.
        Number of iterations.

    '''
    # Test adjacency matrix is OK
    n, _ = G.shape
    assert (G.shape == (n, n))
    # Constants Speed-UP
    deg_out_beta = G.sum(axis=0).T / beta  # vector
    # Initialize
    ranks = np.ones((n, 1)) / n  # vector
    time = 0
    flag = True
    while flag:
        time += 1
        with np.errstate(divide='ignore'):  # Ignore division by 0 on ranks/deg_out_beta
            new_ranks = G.dot((ranks / deg_out_beta))  # vector
        # Leaked PageRank
        new_ranks += (1 - new_ranks.sum()) / n
        # Stop condition
        if np.linalg.norm(ranks - new_ranks, ord=1) <= epsilon:
            flag = False
        ranks = new_ranks
    return ranks / ranks.max()


def get_fake_test():
    return sparse.load_npz(os.path.join("Dataset", "fake_test.npz"))


def divide_train_test(train_old, threshold=0.8):
    msk = np.random.rand(len(train_old)) < threshold
    train = train_old[msk]
    test = train_old[~msk]
    return train, test


def add_dataframe(df, playlist_id, songs_list):
    if type(playlist_id) is list:
        data = pd.DataFrame({"playlist_id": playlist_id, "track_id": songs_list})
    else:
        data = pd.DataFrame({"playlist_id": [playlist_id] * len(songs_list), "track_id": songs_list})
    df = df.append(data)
    return df


def get_tfidf(matrix):
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(matrix)
    if isinstance(tfidf, csr_matrix):
        return tfidf
    else:
        return csr_matrix(tfidf.toarray())


class RS_Data_Loader(object):
    def __init__(self, split_train_test_validation_quota=[0.8, 0.0, 0.2], distr_split=True, top10k=False,
                 all_train=False):

        super(RS_Data_Loader, self).__init__()

        if abs(sum(split_train_test_validation_quota) - 1.0) > 0.001 or len(split_train_test_validation_quota) != 3:
            raise ValueError(
                "RS_Data_Loader: splitTrainTestValidation must be a probability distribution over Train, Test and Validation")

        print("RS_Data_Loader: loading data...")

        self.all_train = all_train
        self.top10k = top10k
        self.distr_split = distr_split
        self.train = pd.read_csv(os.path.join("Dataset", "train.csv"))
        self.train_sequential = pd.read_csv(os.path.join("Dataset", "train_sequential.csv"))
        self.tracks = pd.read_csv(os.path.join("Dataset", "tracks.csv"))
        self.target_playlist = pd.read_csv(os.path.join("Dataset", "target_playlists.csv"))
        self.ICM = None

        if all_train:
            self.UCB_tfidf_artists = self.get_UCM_matrix_artists(train_path=os.path.join("Dataset", "train.csv"))
            # self.UCB_tfidf_album = self.get_UCM_matrix_album(train_path=os.path.join("Dataset", "train.csv"))
            self.URM_train = create_URM_matrix(self.train)
            self.URM_test = get_fake_test()
            self.URM_validation = get_fake_test()
        else:
            if self.distr_split:
                try:
                    self.UCB_tfidf_artists = self.get_UCM_matrix_artists(train_path=os.path.join("Dataset",
                                                                                                 "train_divided_keep_distrib.csv"))
                    # self.UCB_tfidf_album = self.get_UCM_matrix_album(train_path=os.path.join("Dataset", "new_train.csv"))
                    self.URM_train = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_train_keep_distrib.npz"))
                    self.URM_test = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_test_keep_distrib.npz"))
                    self.URM_validation = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_test_keep_distrib.npz"))
                except FileNotFoundError:
                    data_grouped = self.train_sequential.groupby(self.train_sequential.playlist_id).track_id.apply(list)
                    target_plays = self.target_playlist.playlist_id
                    # in datagrouped è una series con la playlist e la lista delle canzoni nella playlist

                    # questi sono due DF vuoti all'inizio
                    train_keep_dist = pd.DataFrame(columns=["playlist_id", "track_id"])
                    test_keep_dist = pd.DataFrame(columns=["playlist_id", "track_id"])

                    for i in data_grouped.keys():
                        if i in target_plays:
                            #per ogni playlist nelle squential
                            line = data_grouped[i]
                            #prendo l'80% della lunghezza
                            len20 = int(len(line) * .8)
                            # le prime 80% canzoni le metto nl dataframe train e le altre nel test
                            train_keep_dist = add_dataframe(train_keep_dist, i, line[:len20])
                            test_keep_dist = add_dataframe(test_keep_dist, i, line[len20:])

                    sequential_playlists = data_grouped.keys()
                    # qua ci sono tutte le playlist con la rispettiva lista di canzoni
                    data_gr_all = self.train.groupby(self.train.playlist_id).track_id.apply(list)

                    to_add_train, to_add_test = [], []
                    to_add_train_ind, to_add_test_ind = [], []
                    for i in data_gr_all.keys():
                        # per ogni canzone
                        if i not in sequential_playlists and i in target_plays:
                            # se non è nelle sequential
                            line = data_gr_all[i]
                            len20 = int(len(line) * .8)
                            # prendo gli indici dell'80 delle canzoni
                            indexes = random.sample(range(0, len(line)), len20)
                            for ind, el in enumerate(line):
                                # per ogni canzone nella playlist
                                if ind in indexes:
                                    # se è negli indici che ho selezionato a random prima la metto nella lista da aggiungere al train
                                    to_add_train_ind.append(i)
                                    to_add_train.append(el)
                                else:
                                    # altrimenti al test
                                    to_add_test_ind.append(i)
                                    to_add_test.append(el)
                    # poi aggiorni i rispettivi df con le canzoni nella lista
                    train_keep_dist = add_dataframe(train_keep_dist, to_add_train_ind, to_add_train)
                    test_keep_dist = add_dataframe(test_keep_dist, to_add_test_ind, to_add_test)

                    # qua dai df con playlist_id track_id creo la matrice csr (non usaimo validation and test)
                    self.URM_train = create_URM_matrix(train_keep_dist)
                    self.URM_test = create_URM_matrix(test_keep_dist)
                    self.URM_validation = create_URM_matrix(test_keep_dist)

                    sparse.save_npz(os.path.join("IntermediateComputations", "URM_train_keep_distrib.npz"),
                                    self.URM_train)
                    sparse.save_npz(os.path.join("IntermediateComputations", "URM_test_keep_distrib.npz"),
                                    self.URM_test)
                    train_keep_dist.to_csv(os.path.join("Dataset", "train_divided_keep_distrib.csv"))
            elif self.top10k:
                try:
                    self.UCB_tfidf_artists = self.get_UCM_matrix_artists(train_path=os.path.join("Dataset",
                                                                                                 "new_train.csv"))
                    # self.UCB_tfidf_album = self.get_UCM_matrix_album(train_path=os.path.join("Dataset", "new_train.csv"))
                    self.URM_train = scipy.sparse.load_npz(os.path.join("IntermediateComputations", "URM_train.npz"))
                    self.URM_test = scipy.sparse.load_npz(os.path.join("IntermediateComputations", "URM_test.npz"))
                    self.URM_validation = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_test.npz"))
                except FileNotFoundError:
                    start_mask = np.asarray([False] * len(self.train))
                    with open(os.path.join("Dataset", 'top10_playlist.pkl'), 'rb') as handle:
                        top10k_playlist = pickle.load(handle)

                    for top_play in top10k_playlist:
                        my_train = self.train[self.train.playlist_id == top_play]
                        to_take = random.sample(list(my_train.index), 10)
                        start_mask[to_take] = True

                    new_train = self.train[~start_mask]
                    new_test = self.train[start_mask]

                    self.URM_train = create_URM_matrix(new_train)
                    self.URM_test = create_URM_matrix(new_test)
                    self.URM_validation = create_URM_matrix(new_test)

                    new_train.to_csv(os.path.join("Dataset", "new_train.csv"))
                    scipy.sparse.save_npz(os.path.join("IntermediateComputations", "URM_train.npz"), self.URM_train)
                    scipy.sparse.save_npz(os.path.join("IntermediateComputations", "URM_test.npz"), self.URM_test)
                    self.UCB_tfidf_artists = self.get_UCM_matrix_artists(train_path=os.path.join("Dataset",
                                                                                                 "new_train.csv"))
                    # here we use the same train and test

            else:
                train, test = divide_train_test(self.train, threshold=0.85)

                self.URM_train = create_URM_matrix(train)
                self.URM_test = create_URM_matrix(test)
                self.URM_validation = self.URM_test
                train.to_csv(os.path.join("Dataset", "new_train_no_top10k.csv"))
                self.UCB_tfidf_artists = self.get_UCM_matrix_artists(
                    train_path=os.path.join("Dataset", "new_train_no_top10k.csv"))
                # URM_all = create_URM_matrix(train)
                # self.URM_train, self.URM_test, self.URM_validation = split_train_validation_test(URM_all,
                #                                                                                  split_train_test_validation_quota)

        print("RS_Data_Loader: loading complete")

    def get_URM_train(self):
        return self.URM_train

    def get_URM_test(self):
        return self.URM_test

    def get_URM_validation(self):
        return self.URM_validation

    def get_target_playlist(self):
        return self.target_playlist

    def get_traks(self):
        return self.tracks

    def get_tfidf_artists(self):
        return self.UCB_tfidf_artists

    def get_tfidf_album(self):
        return self.UCB_tfidf_album

    def get_ICM(self):
        if self.ICM is None:
            self.ICM = get_icm_matrix(self.tracks)
        return self.ICM

    def get_page_rank_URM(self):
        try:
            with open(os.path.join("Dataset", "URM_pagerank.pkl"), 'rb') as handle:
                to_ret = pickle.load(handle)
                # print(to_ret)
                return to_ret
        except FileNotFoundError:
            self.URM_train = self.URM_train.astype(float)
            l = range(self.URM_train.shape[1])
            s_all = set(l)
            relation_mat_gen = self.URM_train.transpose().dot(self.URM_train).tocsc()
            t = time.time()
            URM_new = self.URM_train
            for user_id in range(self.URM_train.shape[0]):
                if user_id % 100 == 0:
                    print(user_id)
                    print("Avg time spent: {}".format((time.time() - t) / 100))
                    t = time.time()
                relation_mat = relation_mat_gen.copy()
                songs_in_playlist = self.URM_train.indices[
                                    self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                s_i = s_all - set(songs_in_playlist)
                for i in s_i:
                    relation_mat.data[relation_mat.indptr[i]:relation_mat.indptr[i + 1]].fill(0)
                relation_mat.eliminate_zeros()
                page_rank = compute_PageRank(relation_mat.transpose()).A1
                # print(type(self.URM_train[user_id].data))
                # URM_new[user_id].data = np.multiply(URM_new[user_id].data, page_rank[songs_in_playlist])
                URM_new[user_id] = URM_new[user_id].multiply(page_rank)
                del relation_mat
            print("URM modified")
            with open(os.path.join("Dataset", "URM_pagerank.pkl"), 'wb') as handle:
                pickle.dump(URM_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return self.URM_train

    def create_complete_test(self):
        row = 50446
        col = 20635
        return csr_matrix(([1] * row, (range(row), [0] * row)), shape=(row, col))

    def get_UCM_matrix_artists(self, train_path=""):
        try:
            if self.all_train:
                with open(os.path.join("Dataset", "UserCBF_artists_all.pkl"), 'rb') as handle:
                    to_ret = pickle.load(handle)
                    return to_ret
            else:
                if self.distr_split:
                    with open(os.path.join("Dataset", "UserCBF_artists.pkl"), 'rb') as handle:
                        to_ret = pickle.load(handle)
                        return to_ret
                else:
                    with open(os.path.join("Dataset", "UserCBF_artists_notop10.pkl"), 'rb') as handle:
                        to_ret = pickle.load(handle)
                        return to_ret

        except FileNotFoundError:
            train = pd.read_csv(train_path)
            tracks_for_playlist = train.merge(self.tracks, on="track_id").loc[:, 'playlist_id':'artist_id'].sort_values(
                by="playlist_id")
            playlists_arr = tracks_for_playlist.playlist_id.unique()
            artists_arr = self.tracks.artist_id.unique()
            UCM_artists = np.zeros(shape=(50446, artists_arr.shape[0]))

            def create_feature_artists(entry):
                if entry.playlist_id in playlists_arr:
                    UCM_artists[entry.playlist_id][entry.artist_id] += 1

            tracks_for_playlist.apply(create_feature_artists, axis=1)
            UCM_tfidf = get_tfidf(UCM_artists)

            if self.all_train:
                print("All train artists matrix saved")
                with open(os.path.join("Dataset", "UserCBF_artists_all.pkl"), 'wb') as handle:
                    pickle.dump(UCM_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                if self.distr_split:
                    with open(os.path.join("Dataset", "UserCBF_artists.pkl"), 'wb') as handle:
                        pickle.dump(UCM_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return UCM_tfidf

    def get_UCM_matrix_album(self, train_path=None):
        try:
            if self.all_train:
                with open(os.path.join("Dataset", "UserCBF_album_all.pkl"), 'rb') as handle:
                    to_ret = pickle.load(handle)
                    return to_ret
            else:
                if self.distr_split:
                    with open(os.path.join("Dataset", "UserCBF_album.pkl"), 'rb') as handle:
                        to_ret = pickle.load(handle)
                        return to_ret
                else:
                    with open(os.path.join("Dataset", "UserCBF_album_random.pkl"), 'rb') as handle:
                        to_ret = pickle.load(handle)
                        return to_ret


        except FileNotFoundError:
            train = pd.read_csv(train_path)
            tracks_for_playlist = train.merge(self.tracks, on="track_id").loc[:, 'playlist_id':'album_id'].sort_values(
                by="playlist_id")
            playlists_arr = tracks_for_playlist.playlist_id.unique()
            album_arr = self.tracks.album_id.unique()
            UCM_albums = np.zeros(shape=(50446, album_arr.shape[0]))

            def create_feature_album(entry):
                UCM_albums[entry.playlist_id][entry.album_id] += 1

            tracks_for_playlist.apply(create_feature_album, axis=1)

            UCM_tfidf = get_tfidf(UCM_albums)

            if self.all_train:
                with open(os.path.join("Dataset", "UserCBF_album_all.pkl"), 'wb') as handle:
                    pickle.dump(UCM_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                if self.distr_split:
                    with open(os.path.join("Dataset", "UserCBF_album.pkl"), 'wb') as handle:
                        pickle.dump(UCM_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(os.path.join("Dataset", "UserCBF_album_random.pkl"), 'wb') as handle:
                        pickle.dump(UCM_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return UCM_tfidf
