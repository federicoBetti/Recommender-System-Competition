#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""
import os

import pickle

import sys

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender

from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sps

from ParameterTuning.BayesianSearch import writeLog


class PureSVDRecommender(Recommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train):
        super(PureSVDRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.compute_item_score = self.compute_score_SVD
        self.U, self.Sigma, self.VT, self.s_Vt = None, None, None, None

    def fit(self, num_factors=100, force_compute_sim=True):

        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("IntermediateComputations", "PureSVDMatrices.pkl"), 'rb') as handle:
                    (U, s_Vt) = pickle.load(handle)
            except FileNotFoundError:
                found = False

            if found:
                self.U = U
                self.s_Vt = s_Vt
                print("Saved Pure SVD Matrices Used!")
                return

        from sklearn.utils.extmath import randomized_svd

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition...")

        self.U, self.Sigma, self.VT = randomized_svd(self.URM_train,
                                                     n_components=num_factors,
                                                     # n_iter=5,
                                                     random_state=None)

        self.s_Vt = sps.diags(self.Sigma) * self.VT

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition... Done!")

        with open(os.path.join("IntermediateComputations", "PureSVDMatrices.pkl"), 'wb') as handle:
            pickle.dump((self.U, self.s_Vt), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

            # truncatedSVD = TruncatedSVD(n_components = num_factors)
            #
            # truncatedSVD.fit(self.URM_train)
            #
            # truncatedSVD

            # U, s, Vt =

    def compute_score_SVD(self, user_id_array):

        try:

            item_weights = self.U[user_id_array, :].dot(self.s_Vt)

        except Exception as e:

            writeLog("PureSVD: Note able to retrieve item weights - Exception {}\n".format(str(e)),
                     self.logFile)
            sys.exit(-1)

        return item_weights

    def saveModel(self, folder_path, file_name=None):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "U": self.U,
            "Sigma": self.Sigma,
            "VT": self.VT,
            "s_Vt": self.s_Vt
        }

        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete")
