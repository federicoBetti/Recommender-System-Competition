#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/11/18

@author: Federico Betti
"""
import time

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
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


class HybridRecommender(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

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

        self.sparse_weights = sparse_weights

        self.recommender_list = []
        self.weights = weights

        self.normalize = None
        self.topK = None
        self.shrink = None

        #TODO: HO MESSO TFIDF PER VEDERE RISULTATO. NEL CASO FACCIA CAGARE TOGLIERE

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
            normalize=True,
            old_similarity_matrix=None, epochs=1,
            force_compute_sim=False, **similarity_args):

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

        assert self.weights is not None, "Weights Are None!"

        assert len(self.recommender_list) == len(self.weights), "Weights and recommender list have different lenghts"

        assert len(topK) == len(shrink) == len(self.recommender_list), "Knns, Shrinks and recommender list have " \
                                                                       "different lenghts "

        self.normalize = normalize
        self.topK = topK
        self.shrink = shrink

        for knn, shrink, recommender in zip(topK, shrink, self.recommender_list):
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

            elif recommender.__class__ in [SLIMElasticNetRecommender]:
                recommender.fit(topK=knn, l1_ratio=similarity_args["l1_ratio"], force_compute_sim=force_compute_sim)

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
            # return [0, 0, 0, 0, 0, 0, 0, 0]
            # return self.d_weights[0]
            return [0.45590938562950867, 0, 0.23905548168035573, 0.017005850670624212, 0.9443556793576228, 0.19081956929601618, 0, 0.11267140391070507]

        elif pop[0] < level < pop[1]:
            # return self.weights
            # return [0, 0, 0, 0, 0, 0, 0, 0]
            return [0.973259052781316, 0, 0.8477517414017691, 0.33288193455193427, 0.9696801027638645, 0.4723616073494711, 0, 0.4188403112229081]
            # return self.d_weights[1]
        else:
            # return self.weights
            # return [0, 0, 0, 0, 0, 0, 0, 0]
            # return self.d_weights[2]
            return [0.9780713488404191, 0, 0.9694246318172682, 0.5703399158380364, 0.9721597253259535, 0.9504112133900943, 0, 0.9034510004379944]

    def compute_score_hybrid(self, recommender, user_id_array, dict_pop, remove_seen_flag=True,
                             remove_top_pop_flag=False,
                             remove_CustomItems_flag=False):
        scores = []
        final_score = np.zeros(len(recommender.recommender_list))
        for rec in recommender.recommender_list:
            if rec.__class__ in [HybridRecommender]:
                scores.append(self.compute_score_hybrid(recommender, user_id_array, dict_pop))
                continue
            scores_batch = rec.compute_item_score(user_id_array)
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

        if recommender.dynamic:
            for user_index in range(len(user_id_array)):
                user_id = user_id_array[user_index]
                user_profile = recommender.URM_train.indices[
                               recommender.URM_train.indptr[user_id]:recommender.URM_train.indptr[user_id + 1]]

                if recommender.onPop:
                    level = int(ged.playlist_popularity(user_profile, dict_pop))
                else:
                    level = int(ged.lenght_playlist(user_profile))
                weights = recommender.change_weights(level, recommender.pop)
                final_score_line = np.zeros(scores[0].shape[1])
                for score, weight in zip(scores, weights):
                    final_score_line += (score[user_index] * weight)
                final_score[user_index] = final_score_line
        else:
            for score, weight in zip(scores, recommender.weights):
                final_score += (score * weight)
        return final_score

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

            if self.dynamic:
                for user_index in range(len(user_id_array)):
                    user_id = user_id_array[user_index]
                    user_profile = self.URM_train.indices[
                                   self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                    if self.onPop:
                        level = int(ged.playlist_popularity(user_profile, dict_pop))
                    else:
                        level = int(ged.lenght_playlist(user_profile))
                    weights = self.change_weights(level, self.pop)
                    assert len(weights) == len(scores), "Scores and weights have different lengths"

                    final_score_line = np.zeros(scores[0].shape[1])
                    if sum(weights) > 0:
                        for score, weight in zip(scores, weights):
                            final_score_line += score[user_index] * weight
                    final_score[user_index] = final_score_line
            else:
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
