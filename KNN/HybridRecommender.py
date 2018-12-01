#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/11/18

@author: Federico Betti
"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import numpy as np

from Base.Similarity.Compute_Similarity import Compute_Similarity
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import Support_functions.get_evaluate_data as ged
from KNN.UserKNNCBFRecommender import UserKNNCBRecommender


class HybridRecommender(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, ICM, recommender_list, UCM_train=None, dynamic=False, d_weights=None, weights=None,
                 URM_validation=None, sparse_weights=True):
        super(Recommender, self).__init__()

        # CSR is faster during evaluation
        self.pop = None
        self.UCM_train = UCM_train
        self.URM_train = check_matrix(URM_train, 'csr')
        self.URM_validation = URM_validation
        self.dynamic = dynamic
        self.d_weights = d_weights
        self.dataset = None

        self.sparse_weights = sparse_weights

        self.recommender_list = []
        self.weights = weights

        self.normalize = None
        self.topK = None
        self.shrink = None

        for recommender in recommender_list:
            if recommender in [SLIM_BPR_Cython, MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython,
                               MatrixFactorization_AsySVD_Cython]:
                print("class recognized")

                self.recommender_list.append(recommender(URM_train, URM_validation=URM_validation))
            elif recommender is ItemKNNCBFRecommender:
                self.recommender_list.append(recommender(ICM, URM_train))
            elif recommender in [PureSVDRecommender]:
                self.recommender_list.append(recommender(URM_train))
            elif recommender in [UserKNNCBRecommender]:
                self.recommender_list.append(recommender(URM_train, UCM_train=self.UCM_train))
            else:  # UserCF, ItemCF, ItemCBF, P3alpha, RP3beta
                self.recommender_list.append(recommender(URM_train))

    def fit(self, topK=None, shrink=None, weights=None, pop=None, weights1=None, weights2=None, weights3=None,
            weights4=None,
            weights5=None, weights6=None, pop1=None, pop2=None, similarity='cosine', normalize=True,
            old_similarity_matrix=None, epochs=1,
            force_compute_sim=False, **similarity_args):

        if self.weights is None:
            if weights is None:
                weights = [weights1, weights2, weights3, weights4, weights5, weights6]
                weights = [x for x in weights if x is not None]
            self.weights = weights

        if self.pop is None:
            if pop is None:
                pop = [pop1, pop2]
                pop = [x for x in pop if x is not None]
            self.pop = pop

        assert self.weights is not None, "Weights Are None!"

        assert len(self.recommender_list) == len(self.weights), "Weights and recommender list have different lenghts"

        assert len(topK) == len(shrink) == len(self.recommender_list), "Knns, Shrinks and recommender list have " \
                                                                       "different lenghts "

        self.normalize = normalize
        self.topK = topK
        self.shrink = shrink

        for knn, shrink, recommender in zip(topK, shrink, self.recommender_list):
            print("FIT")

            if recommender.__class__ is SLIM_BPR_Cython:
                if "lambda_i" in list(similarity_args.keys()):  # lambda i and j provided in args
                    recommender.fit(old_similarity_matrix=old_similarity_matrix, epochs=epochs,
                                    force_compute_sim=force_compute_sim, topK=knn, lambda_i=similarity_args["lambda_i"],
                                    lambda_j=similarity_args["lambda_j"])
                else:
                    recommender.fit(old_similarity_matrix=old_similarity_matrix, epochs=epochs,
                                    force_compute_sim=force_compute_sim, topK=knn)

            elif recommender.__class__ in [MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython,
                                           MatrixFactorization_AsySVD_Cython]:
                recommender.fit(epochs=epochs, force_compute_sim=force_compute_sim)

            elif recommender.__class__ in [PureSVDRecommender]:
                recommender.fit(num_factors=similarity_args["num_factors"], force_compute_sim=force_compute_sim)


            elif recommender.__class__ in [P3alphaRecommender]:
                recommender.fit(topK=knn, alpha=similarity_args["alphaP3"], min_rating=0, implicit=True,
                                normalize_similarity=True, force_compute_sim=force_compute_sim)

            elif recommender.__class__ in [RP3betaRecommender]:
                recommender.fit(alpha=similarity_args["alphaRP3"], beta=similarity_args["betaRP"], min_rating=0,
                                topK=knn, implicit=True, normalize_similarity=True, force_compute_sim=force_compute_sim)

            else:  # ItemCF, UserCF, ItemCBF, UserCBF
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim)

    def change_weights(self, level, pop):
        if level < pop[0]:
            # return [0, 0, 0, 0, 0, 0]
            # return self.weights
            return [0.4, 0.03863232277574469, 0.008527738266632112, 0.2560912624445676, 0.7851755932819731,
                    0.4112843940329439]

        elif pop[0] < level < pop[1]:
            # return [0, 0, 0, 0, 0, 0]
            return [0.2, 0.012499871230102988, 0.020242981888115352, 0.9969708006657074, 0.9999132876156388,
                    0.6888103295594851]

        else:
            # return self.weights
            # return [0, 0, 0, 0, 0, 0]
            return [0.2, 0.10389111810225915, 0.14839466129917822, 0.866992903043857, 0.07010619211847613,
                    0.5873532658846817]

    def recommend(self, user_id_array, dict_pop=None, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False):

        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        weights = self.weights
        if cutoff == None:
            # noinspection PyUnresolvedReferences
            cutoff = self.URM_train.shape[1] - 1
        else:
            cutoff

        # compute the scores using the dot product
        # noinspection PyUnresolvedReferences
        if self.sparse_weights:
            scores = []
            user_profile = self.URM_train[user_id_array]
            # noinspection PyUnresolvedReferences
            for recommender in self.recommender_list:
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
            if self.dynamic:
                for user_index in range(len(user_id_array)):
                    user_id = user_id_array[user_index]
                    user_profile = self.URM_train.indices[
                                   self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                    level = int(ged.lenght_playlist(user_profile))
                    weights = self.change_weights(level, self.pop)
                    # print(weights)
                    final_score_line = np.zeros(scores[0].shape[1])
                    for score, weight in zip(scores, weights):
                        final_score_line += (score[user_index] * weight)
                    final_score[user_index] = final_score_line
            else:
                for score, weight in zip(scores, weights):
                    final_score += (score * weight)

        else:
            raise NotImplementedError

        scores = final_score
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
