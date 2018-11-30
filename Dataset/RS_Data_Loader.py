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
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle


def split_train_validation_test(URM_all, split_train_test_validation_quota):
    URM_all = URM_all.tocoo()
    URM_shape = URM_all.shape

    numInteractions = len(URM_all.data)

    split = np.random.choice([1, 2, 3], numInteractions, p=split_train_test_validation_quota)

    trainMask = split == 1
    URM_train = sps.coo_matrix((URM_all.data[trainMask], (URM_all.row[trainMask], URM_all.col[trainMask])),
                               shape=URM_shape)
    URM_train = URM_train.tocsr()

    testMask = split == 2

    URM_test = sps.coo_matrix((URM_all.data[testMask], (URM_all.row[testMask], URM_all.col[testMask])), shape=URM_shape)
    URM_test = URM_test.tocsr()

    validationMask = split == 3

    URM_validation = sps.coo_matrix(
        (URM_all.data[validationMask], (URM_all.row[validationMask], URM_all.col[validationMask])), shape=URM_shape)
    URM_validation = URM_validation.tocsr()

    return URM_train, URM_validation, URM_test


def create_URM_matrix(train):
    row = list(train.playlist_id)
    col = list(train.track_id)
    return csr_matrix(([1] * len(row), (row, col)), shape=(50446, 20635))


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


def create_complete_test():
    row = 50446
    col = 20635
    return csr_matrix(([1] * row, (range(row), [0] * row)), shape=(row, col))


def get_fake_test():
    return sparse.load_npz(os.path.join("Dataset", "fake_test.npz"))


def divide_train_test(train_old, threshold=0.8):
    msk = np.random.rand(len(train_old)) < threshold
    train = train_old[msk]
    test = train_old[~msk]
    return train, test


class RS_Data_Loader(object):

    def __init__(self, slim_after_hybrid=False, split_train_test_validation_quota=[0.8, 0.0, 0.2], top10k=True, all_train=False):

        super(RS_Data_Loader, self).__init__()

        if abs(sum(split_train_test_validation_quota) - 1.0) > 0.001 or len(split_train_test_validation_quota) != 3:
            raise ValueError(
                "RS_Data_Loader: splitTrainTestValidation must be a probability distribution over Train, Test and Validation")

        print("RS_Data_Loader: loading data...")

        train = pd.read_csv(os.path.join("Dataset", "train.csv"))

        if all_train:
            self.URM_train = create_URM_matrix(train)
            self.URM_test = get_fake_test()
            self.URM_validation = get_fake_test()
        else:
            if top10k:
                try:
                    self.URM_train = scipy.sparse.load_npz(os.path.join("IntermediateComputations", "URM_train.npz"))
                    self.URM_test = scipy.sparse.load_npz(os.path.join("IntermediateComputations", "URM_test.npz"))
                    self.URM_validation = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_test.npz"))
                except FileNotFoundError:

                    start_mask = np.asarray([False] * len(train))
                    with open(os.path.join("Dataset", 'top10_playlist.pkl'), 'rb') as handle:
                        top10k_playlist = pickle.load(handle)

                    for top_play in top10k_playlist:
                        my_train = train[train.playlist_id == top_play]
                        to_take = random.sample(list(my_train.index), 10)
                        start_mask[to_take] = True

                    new_train = train[~start_mask]
                    new_test = train[start_mask]

                    self.URM_train = create_URM_matrix(new_train)
                    self.URM_test = create_URM_matrix(new_test)
                    self.URM_validation = create_URM_matrix(new_test)

                    scipy.sparse.save_npz(os.path.join("IntermediateComputations", "URM_train.npz"), self.URM_train)
                    scipy.sparse.save_npz(os.path.join("IntermediateComputations", "URM_test.npz"), self.URM_test)

                # here we use the same train and test

            else:
                train, test = divide_train_test(train, threshold=0.85)

                self.URM_train = create_URM_matrix(train)
                self.URM_test = create_URM_matrix(test)
                self.URM_validation = self.URM_test
                # URM_all = create_URM_matrix(train)
                # self.URM_train, self.URM_test, self.URM_validation = split_train_validation_test(URM_all,
                #                                                                                  split_train_test_validation_quota)

        # # to delete in case!
        # all_playlist_to_predict = pd.DataFrame(index=train.playlist_id.unique())
        # s = train.playlist_id.unique()
        self.tracks = pd.read_csv(os.path.join("Dataset", "tracks.csv"))
        self.target_playlist = pd.read_csv(os.path.join("Dataset", "target_playlists.csv"))
        # all_train = train.copy()
        # train, test = divide_train_test(train, threshold=0.85)
        # train, validation = divide_train_test(train, threshold=0.85)
        # test_songs_per_playlist = make_series(test)
        # all_playlist_to_predict['playlist_id'] = pd.Series(range(len(all_playlist_to_predict.index)),
        #                                                    index=all_playlist_to_predict.index)
        self.ICM = get_icm_matrix(self.tracks)

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

    def get_ICM(self):
        # todo save ICM in a file, since it is always the same
        if self.ICM is None:
            self.ICM = get_icm_matrix(self.tracks)
        return self.ICM
