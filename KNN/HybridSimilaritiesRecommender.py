#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/11/18

@author: Federico Betti
"""
import pickle
import os
import time

from scipy.sparse import csr_matrix

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import numpy as np

from Base.Similarity.Compute_Similarity import Compute_Similarity
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN import HybridRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import Support_functions.get_evaluate_data as ged
from KNN.UserKNNCBFRecommender import UserKNNCBRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


class HybridSimilaritiesRecommender(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "HybridSimilaritiesRecommender"

    def __init__(self, URM_train, ICM, recommender_list, UCM_train=None, dynamic=False, d_weights=None, weights=None,
                 URM_validation=None, sparse_weights=True, onPop=True, moreHybrids=False):
        super(Recommender, self).__init__()

        # CSR is faster during evaluation
        self.pop = None
        self.UCM_train = UCM_train
        self.URM_train = check_matrix(URM_train, 'csr')
        self.URM_validation = URM_validation
        self.dynamic = dynamic
        self.d_weights = d_weights
        self.dataset = None
        self.onPop = onPop
        self.moreHybrids = moreHybrids
        # with open(os.path.join("Dataset", "Cluster_0_dict_Kmeans_3.pkl"), 'rb') as handle:
        #     self.cluster_0 = pickle.load(handle)
        # with open(os.path.join("Dataset", "Cluster_1_dict_Kmeans_3.pkl"), 'rb') as handle:
        #     self.cluster_1 = pickle.load(handle)
        # with open(os.path.join("Dataset", "Cluster_2_dict_Kmeans_3.pkl"), 'rb') as handle:
        #     self.cluster_2 = pickle.load(handle)

        self.sparse_weights = sparse_weights

        self.recommender_list = []
        self.weights = weights

        self.normalize = None
        self.topK = None
        self.shrink = None

        self.recommender_number = len(recommender_list)
        if self.d_weights is None:
            # 3 because we have divided in 3 intervals
            self.d_weights = [[0] * self.recommender_number, [0] * self.recommender_number,
                              [0] * self.recommender_number]

        for recommender in recommender_list:
            if recommender in [SLIM_BPR_Cython, MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython,
                               MatrixFactorization_AsySVD_Cython]:
                print("class recognized")

                self.recommender_list.append(recommender(URM_train, URM_validation=URM_validation))
            elif recommender is ItemKNNCBFRecommender:
                self.recommender_list.append(recommender(ICM, URM_train))
            elif recommender in [PureSVDRecommender, SLIMElasticNetRecommender]:
                self.recommender_list.append(recommender(URM_train))
            elif recommender in [UserKNNCBRecommender]:
                self.recommender_list.append(recommender(self.UCM_train, URM_train))
            # For sake of simplicity the recommender in this case is initialized and fitted outside
            elif recommender.__class__ in [HybridRecommender]:
                self.recommender_list.append(recommender)

            else:  # UserCF, ItemCF, ItemCBF, P3alpha, RP3beta
                self.recommender_list.append(recommender(URM_train))

    def fit(self, topK=None, shrink=None, weights=None, pop=None, weights1=None, weights2=None, weights3=None,
            weights4=None,
            weights5=None, weights6=None, weights7=None, weights8=None, pop1=None, pop2=None, similarity='cosine',
            normalize=True, final_weights=None, final_weights1=None, final_weights2=None,
            old_similarity_matrix=None, epochs=1, top1=None, shrink1=None,
            force_compute_sim=False, weights_to_dweights=-1, **similarity_args):

        if topK is None:  # IT MEANS THAT I'M TESTING ONE RECOMMENDER ON A SPECIIFC INTERVAL
            topK = [top1]
            shrink = [shrink1]

        if self.weights is None:
            if weights1 is not None:
                weights = [weights1, weights2, weights3, weights4, weights5, weights6, weights7, weights8]
                weights = [x for x in weights if x is not None]
            self.weights = weights

        if self.pop is None:
            if pop is None:
                pop = [pop1, pop2]
                pop = [x for x in pop if x is not None]
            self.pop = pop

        if weights_to_dweights != -1:
            self.d_weights[weights_to_dweights] = self.weights

        assert self.weights is not None, "Weights Are None!"

        assert len(self.recommender_list) == len(
            self.weights), "Weights: {} and recommender list: {} have different lenghts".format(len(self.weights), len(
            self.recommender_list))

        assert len(topK) == len(shrink) == len(self.recommender_list), "Knns, Shrinks and recommender list have " \
                                                                       "different lenghts "

        self.normalize = normalize
        self.topK = topK
        self.shrink = shrink

        self.gradients = [0] * self.recommender_number
        self.MAE = 0
        p3counter = 0
        rp3bcounter = 0
        slim_counter = 0
        factorCounter = 0

        for knn, shrink, recommender in zip(topK, shrink, self.recommender_list):
            if recommender.__class__ is SLIM_BPR_Cython:
                if "lambda_i" in list(similarity_args.keys()):  # lambda i and j provided in args
                    if type(similarity_args["lambda_i"]) is not list:
                        similarity_args["lambda_i"] = [similarity_args["lambda_i"]]
                        similarity_args["lambda_j"] = [similarity_args["lambda_j"]]
                    recommender.fit(old_similarity_matrix=old_similarity_matrix, epochs=epochs,
                                    force_compute_sim=force_compute_sim, topK=knn,
                                    lambda_i=similarity_args["lambda_i"][slim_counter],
                                    lambda_j=similarity_args["lambda_j"][slim_counter])
                else:
                    recommender.fit(old_similarity_matrix=old_similarity_matrix, epochs=epochs,
                                    force_compute_sim=force_compute_sim, topK=knn)
                slim_counter += 1

            elif recommender.__class__ in [MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython,
                                           MatrixFactorization_AsySVD_Cython]:
                recommender.fit(epochs=epochs, force_compute_sim=force_compute_sim)

            elif recommender.__class__ in [SLIMElasticNetRecommender]:
                recommender.fit(topK=knn, l1_ratio=similarity_args["l1_ratio"], force_compute_sim=force_compute_sim)

            elif recommender.__class__ in [PureSVDRecommender]:
                recommender.fit(num_factors=similarity_args["num_factors"][factorCounter],
                                force_compute_sim=force_compute_sim)
                factorCounter += 1

            elif recommender.__class__ in [P3alphaRecommender]:
                if type(similarity_args["alphaP3"]) is not list:
                    similarity_args["alphaP3"] = [similarity_args["alphaP3"]]
                recommender.fit(topK=knn, alpha=similarity_args["alphaP3"][p3counter], min_rating=0, implicit=True,
                                normalize_similarity=True, force_compute_sim=force_compute_sim)
                p3counter += 1

            elif recommender.__class__ in [RP3betaRecommender]:
                recommender.fit(alpha=similarity_args["alphaRP3"][rp3bcounter],
                                beta=similarity_args["betaRP"][rp3bcounter], min_rating=0,
                                topK=knn, implicit=True, normalize_similarity=True, force_compute_sim=force_compute_sim)
                rp3bcounter += 1

            elif recommender.__class__ in [ItemKNNCBFRecommender]:
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim,
                                feature_weighting_index=similarity_args["feature_weighting_index"])

            else:  # ItemCF, UserCF, ItemCBF, UserCBF
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim)

        print("Recommender list before: {}".format(self.recommender_list))
        self.W_sparse20 = csr_matrix(([], ([], [])), shape=(20635, 20635))
        self.W_sparse50 = csr_matrix(([], ([], [])), shape=(50446, 50446))
        to_delete = []
        for index, recommender in enumerate(self.recommender_list):
            try:
                self.W_sparse20 += self.weights[index] * recommender.W_sparse
                to_delete.append(recommender)
                print("Recommender {} is summed in W_sparse20".format(recommender))
                continue
            except:
                # the recommender doesn't have a W sparse matrix of that shape
                a = 1

            try:
                self.W_sparse50 += self.weights[index] * recommender.W_sparse
                to_delete.append(recommender)
                print("Recommender {} is summed in W_sparse50".format(recommender))
                continue
            except:
                # the recommender doesn't have a W sparse matrix of that shape
                a = 1

        # remove recommenders that already has the similarity merged
        self.recommender_list = [x for x in self.recommender_list if x not in to_delete]

        print("Recommender list after: {}".format(self.recommender_list))
        new_item_recommender = ItemKNNCFRecommender(self.URM_train)
        new_item_recommender.W_sparse = self.W_sparse20
        self.recommender_list.append(new_item_recommender)

        new_user_recommender = UserKNNCFRecommender(self.URM_train)
        new_user_recommender.W_sparse = self.W_sparse50
        self.recommender_list.append(new_user_recommender)
        print("Recommender list final: {}".format(self.recommender_list))

        if final_weights is None:
            self.final_weights = [final_weights1, final_weights2]
        else:
            self.final_weights = final_weights

        assert len(final_weights) == len(
            self.recommender_list), "Lunghezza di final weights e dei reccomender rimasti è diversa. Sono rimasti {} recommender con i final weight di lunghezza {}".format(
            len(self.recommender_list), len(final_weights))

    def recommend(self, user_id_array, dict_pop=None, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False):

        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        weights = self.final_weights
        if cutoff == None:
            # noinspection PyUnresolvedReferences
            cutoff = self.URM_train.shape[1] - 1
        else:
            cutoff

        # compute the scores using the dot product
        # noinspection PyUnresolvedReferences
        if self.sparse_weights:
            scores = []
            # noinspection PyUnresolvedReferences
            for recommender in self.recommender_list:
                if recommender.__class__ in [HybridRecommender]:
                    scores.append(self.compute_score_hybrid(recommender, user_id_array, dict_pop,
                                                            remove_seen_flag=True, remove_top_pop_flag=False,
                                                            remove_CustomItems_flag=False))
                    continue
                scores_batch = recommender.compute_item_score(user_id_array)
                # scores_batch = np.ravel(scores_batch) # because i'm not using batch

                for user_index in range(len(user_id_array)):

                    user_id = user_id_array[user_index]

                    if remove_seen_flag:
                        scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

                if remove_top_pop_flag:
                    scores_batch = self._remove_TopPop_on_scores(scores_batch)

                if remove_CustomItems_flag:
                    scores_batch = self._remove_CustomItems_on_scores(scores_batch)

                scores.append(scores_batch)

            final_score = np.zeros(scores[0].shape)

            for score, weight in zip(scores, weights):
                    final_score += (score * weight)

        else:
            raise NotImplementedError

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-final_score).argpartition(cutoff, axis=1)[:, 0:cutoff]

        relevant_items_partition_original_value = final_score[
            np.arange(final_score.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        # scores = final_score
        # # relevant_items_partition is block_size x cutoff
        # relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]
        #
        # relevant_items_partition_original_value = scores_batch[
        #     np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        # relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        # ranking = relevant_items_partition[
        #     np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking
