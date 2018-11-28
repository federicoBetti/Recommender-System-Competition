#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/11/18

"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import numpy as np

from Base.Similarity.Compute_Similarity import Compute_Similarity
from KNN.HybridRecommender import HybridRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import Support_functions.get_evaluate_data as ged


class HybridRecommenderTopNapproach(HybridRecommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "HybridRecommenderTopNapproach"

    def __init__(self, URM_train, ICM, recommeder_list, d_weights=None, dynamic=False, weights=None,
                 URM_validation=None, sparse_weights=True):
        super(Recommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')
        self.URM_validation = URM_validation
        self.dynamic = dynamic
        self.dataset = None
        self.d_weights = d_weights
        self.sparse_weights = sparse_weights

        self.recommender_list = []
        self.weights = weights

        for recommender in recommeder_list:
            if recommender in [SLIM_BPR_Cython, MatrixFactorization_BPR_Cython]:
                print("class recognized")
                self.recommender_list.append(recommender(URM_train, URM_validation=URM_validation))
            elif recommender is ItemKNNCBFRecommender:
                self.recommender_list.append(recommender(ICM, URM_train))
            else:
                self.recommender_list.append(recommender(URM_train))

    def change_weights(self, level, pop):
        if level < pop[0]:
            return self.d_weights[0]

        elif pop[0] < level < pop[1]:
            return self.d_weights[1]

        elif pop[1] < level < pop[2]:
            return self.d_weights[2]

        else:
            return self.d_weights[3]

    # topk1,2,3 e shrink e weights1,2,3 sono quelli del dizionario, aggiungerli per il test
    def fit(self, topK=None, shrink=None, weights=None, topK1=None, topK2=None, topK3=None, shrink1=None, shrink2=None,
            shrink3=None, weights1=None, weights2=None, weights3=None, similarity='cosine', normalize=True,
            old_similrity_matrix=None, epochs=1, force_compute_sim=False, **similarity_args):

        if shrink is None:
            shrink = [shrink1, shrink2, shrink3]
            shrink = [x for x in shrink if x is not None]
        if topK is None:
            topK = [topK1, topK2, topK3]
            topK = [x for x in topK if x is not None]
        if weights is None:
            weights = [weights1, weights2, weights3]
            weights = [x for x in weights if x is not None]

        if self.weights is None:
            self.weights = weights

        assert self.weights is not None, "Weights Are None!"

        assert len(self.recommender_list) == len(self.weights), "Weights and recommender list have different lenghts"

        assert len(topK) == len(shrink) == len(self.recommender_list), "Knns, Shrinks and recommender list have " \
                                                                       "different lenghts "

        self.normalize = normalize
        self.topK = topK
        self.shrink = shrink

        for knn, shrink, recommender in zip(topK, shrink, self.recommender_list):
            if recommender.__class__ is SLIM_BPR_Cython:
                recommender.fit(old_similrity_matrix=old_similrity_matrix, epochs=epochs,
                                force_compute_sim=force_compute_sim)

            elif recommender.__class__ is MatrixFactorization_BPR_Cython:
                recommender.fit(epochs=epochs, force_compute_sim=force_compute_sim)

            else:  # ItemCF, UserCF, ItemCBF
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim)

    def add_non_present_items(self, final_ranking, current_ranking, weight, n):
        count = 0
        for item in current_ranking:
            if item in final_ranking:
                continue
            final_ranking.append(item)
            count += 1
            if count == weight or len(final_ranking) == n:
                return final_ranking
        return final_ranking

    def fill_missing_items(self, final_ranking, rankings, n):
        for ranking in rankings:
            final_ranking = self.add_non_present_items(final_ranking, ranking, 10, n)
        return final_ranking

    def recommend(self, user_id, dict_pop=None, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False):

        weights = self.weights
        if cutoff is None:
            # noinspection PyUnresolvedReferences
            n = self.URM_train.shape[1] - 1
        else:
            n = cutoff

        # compute the scores using the dot product
        # noinspection PyUnresolvedReferences
        if self.sparse_weights:
            scores = []
            user_profile = self.URM_train[user_id]
            # noinspection PyUnresolvedReferences
            for recommender in self.recommender_list:
                scores_batch = recommender.compute_item_score(user_id)

                if remove_seen_flag:
                    scores_batch = self._remove_seen_on_scores(user_id, scores_batch)

                if remove_top_pop_flag:
                    scores_batch = self._remove_TopPop_on_scores(scores_batch)

                if remove_CustomItems_flag:
                    scores_batch = self._remove_CustomItems_on_scores(scores_batch)

                scores.append(scores_batch)

            try:
                assert (len(scores) == len(weights))
            except:
                print("Weights and scores from similarities have two different lenghts: {} and {}".format(len(scores),
                                                                                                          len(weights)))
                raise TypeError

            # Weight is the number of items to extract for each ranking
            final_ranking = []
            rankings = []

            if self.dynamic:
                pop = [100, 200]
                user_profile_pop = self.URM_train.indices[
                                   self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                level = int(ged.playlist_popularity(user_profile_pop, dict_pop))
                weights = self.change_weights(level, pop)

                # needed since we have to take first the more important recommendations from more important recommender
                # if we don't reach the aimed number of songs
                sorted_scores = [x for _, x in sorted(zip(weights, scores), key=lambda pair: pair[0])]
                weights.sort(reverse=True)
                scores = sorted_scores

            for score, weight in zip(scores, weights):
                relevant_items_partition = (-score).argpartition(n)[0:n]
                relevant_items_partition_sorting = np.argsort(-score[relevant_items_partition])
                ranking = relevant_items_partition[relevant_items_partition_sorting]
                rankings.append(ranking)
                final_ranking = self.add_non_present_items(final_ranking, ranking, weight, n)

            if len(final_ranking) != n:
                final_ranking = self.fill_missing_items(final_ranking, rankings, n)

        else:
            raise NotImplementedError

            user_profile = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            user_ratings = self.URM_train.data[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

            relevant_weights = self.W[user_profile]
            scores = relevant_weights.T.dot(user_ratings)

        return final_ranking

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self.compute_item_score(user_id_array)

        # if self.normalize:
        #     # normalization will keep the scores in the same range
        #     # of value of the ratings in dataset
        #     user_profile = self.URM_train[user_id]
        #
        #     rated = user_profile.copy()
        #     rated.data = np.ones_like(rated.data)
        #     if self.sparse_weights:
        #         den = rated.dot(self.W_sparse).toarray().ravel()
        #     else:
        #         den = rated.dot(self.W).ravel()
        #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
        #     scores /= den

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
                # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
                # ranking = relevant_items_partition[relevant_items_partition_sorting]
                #
                # ranking_list.append(ranking)

        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_CustomItems_flag:
            scores_batch = self._remove_CustomItems_on_scores(scores_batch)

        # scores_batch = np.arange(0,3260).reshape((1, -1))
        # scores_batch = np.repeat(scores_batch, 1000, axis = 0)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[
            np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking_list
