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
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import Support_functions.get_evaluate_data as ged


class HybridRecommender(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, ICM, recommeder_list, dynamic=False, d_weights=None, weights=None,
                 URM_validation=None, sparse_weights=True):
        super(Recommender, self).__init__()

        # CSR is faster during evaluation
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

        for recommender in recommeder_list:
            if recommender in [SLIM_BPR_Cython, MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython,
                               MatrixFactorization_AsySVD_Cython]:
                print("class recognized")
                self.recommender_list.append(recommender(URM_train, URM_validation=URM_validation))
            elif recommender is ItemKNNCBFRecommender:
                self.recommender_list.append(recommender(ICM, URM_train))
            else:
                self.recommender_list.append(recommender(URM_train))

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

            elif recommender.__class__ in [MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython,
                                           MatrixFactorization_AsySVD_Cython]:
                recommender.fit(epochs=epochs, force_compute_sim=force_compute_sim)

            else:  # ItemCF, UserCF, ItemCBF
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim)

    def recommend(self, user_id, dict_pop=None, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False):

        weights = self.weights
        if cutoff == None:
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

            # QUA È DOVE VENGONO APPLICATI I WEIGHTS AGLI SCORE, QUINDI NEL CASO SI VOLESSE MODFICIARE È QUA!!
            final_score = np.zeros(scores[0].shape)
            if self.dynamic:
                pop = [150, 400, 575]
                user_profile_pop = self.URM_train.indices[
                                   self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                threshold = int(ged.playlist_popularity(user_profile_pop, dict_pop))
                weights = self.change_weights(threshold, pop)

            for score, weight in zip(scores, weights):
                print(score[:30])
                final_score += (score * weight)

        else:
            raise NotImplementedError
            #
            # user_profile = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            # user_ratings = self.URM_train.data[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            #
            # relevant_weights = self.W[user_profile]
            # scores = relevant_weights.T.dot(user_ratings)

        scores = final_score

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        return ranking

    def change_weights(self, level, pop):
        if level < pop[0]:
            return self.d_weights[0]

        elif pop[0] < level < pop[1]:
            return self.d_weights[1]

        elif pop[1] < level < pop[2]:
            return self.d_weights[2]

        else:
            return self.d_weights[3]
