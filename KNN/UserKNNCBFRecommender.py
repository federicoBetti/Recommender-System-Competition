#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""
import os
import pickle
import numpy as np

from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender

from Base.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCBRecommender(SimilarityMatrixRecommender, Recommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCBRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, UCM_train, URM_train, sparse_weights=True):
        super(UserKNNCBRecommender, self).__init__()

        # Not sure if CSR here is faster
        self.URM_train = check_matrix(URM_train, 'csr')
        self.UCM_train = check_matrix(UCM_train, 'csr')
        self.dataset = None

        self.sparse_weights = sparse_weights

        self.compute_item_score = self.compute_score_user_based

    def fit(self, topK=350, shrink=10, similarity='cosine', normalize=True, force_compute_sim=True,
            feature_weighting_index=2, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        self.feature_weighting_index = feature_weighting_index
        feature_weighting = self.FEATURE_WEIGHTING_VALUES[feature_weighting_index]
        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("IntermediateComputations", "UCB",
                                       "totURM={}_totUCM={}_topK={}_shrink={}_feat_wei={}.pkl".format(
                                           str(len(self.URM_train.data)),
                                           str(len(self.UCM_train.data)),
                                           str(self.topK),
                                           str(self.shrink), str(self.feature_weighting_index))),
                          'rb') as handle:
                    W_sparse_new = pickle.load(handle)
            except FileNotFoundError:
                found = False

            if found:
                self.W_sparse = W_sparse_new
                print("Saved User Content Base Similarity Matrix Used!")
                return

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = okapi_BM_25(self.UCM_train)

        elif feature_weighting == "TF-IDF":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = TF_IDF(self.UCM_train)

        similarity = Compute_Similarity(self.UCM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
            with open(os.path.join("IntermediateComputations", "UCB",
                                   "totURM={}_totUCM={}_topK={}_shrink={}_feat_wei={}.pkl".format(
                                       str(len(self.URM_train.data)),
                                       str(len(self.UCM_train.data)),
                                       str(self.topK),
                                       str(self.shrink), str(self.feature_weighting_index))),
                      'wb') as handle:
                pickle.dump(self.W_sparse, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()
