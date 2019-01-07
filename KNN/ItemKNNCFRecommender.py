#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""
import os
import pickle

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Dataset.RS_Data_Loader import RS_Data_Loader, get_tfidf

from Support_functions import get_evaluate_data as ged
from Base.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCFRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(ItemKNNCFRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

        self.W_sparse = None
        self.topK = None
        self.shrink = None
        self.tfidf = None

    def __str__(self):
        return "Item Collaborative Filterng (tokK={}, shrink={}, tfidf={}, normalize={}".format(
            self.topK, self.shrink, self.tfidf, self.normalize)

    def fit(self, topK=350, shrink=10, similarity='cosine', normalize=True, force_compute_sim=True, tfidf=True,
            **similarity_args):

        self.topK = topK
        self.shrink = shrink
        self.tfidf = tfidf

        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("IntermediateComputations", "ItemCFMatrix.pkl"), 'rb') as handle:
                    (topK_new, shrink_new, W_sparse_new) = pickle.load(handle)
            except FileNotFoundError:
                print("File {} not found".format(os.path.join("IntermediateComputations", "ItemCFMatrix.pkl")))
                found = False

            if found and self.topK == topK_new and self.shrink == shrink_new:
                self.W_sparse = W_sparse_new
                print("Saved Item CF Similarity Matrix Used!")
                return

        if tfidf:
            sim_matrix_pre = get_tfidf(self.URM_train)
        else:
            sim_matrix_pre = self.URM_train

        similarity = Compute_Similarity(sim_matrix_pre, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)
        print('Similarity item based CF computed')

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
            with open(os.path.join("IntermediateComputations", "ItemCFMatrix.pkl"), 'wb') as handle:
                pickle.dump((self.topK, self.shrink, self.W_sparse), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Item CF similarity matrix saved")
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()
