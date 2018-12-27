#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import numpy as np
from Dataset.RS_Data_Loader import RS_Data_Loader
from scipy import sparse

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
import bisect


def get_top_items(newTrainXGBoost, songs_in_playlist):
    songs_in_train_in_play = [x for x in newTrainXGBoost if x[0] in songs_in_playlist]
    songs_in_train_in_play = np.asarray(songs_in_train_in_play)
    songs_not_in_train_in_play = [x for x in newTrainXGBoost if x[0] not in songs_in_playlist]
    songs_not_in_train_in_play = np.asarray(songs_not_in_train_in_play)

    user_profile = songs_in_train_in_play.mean(axis=0)

    from scipy.spatial import distance
    distances = [distance.euclidean(s, user_profile) for s in songs_not_in_train_in_play]
    distances = np.asarray(distances)
    ind_max = np.argsort(-distances)[:10]
    distances_sorted = songs_not_in_train_in_play[ind_max]
    return [x[0] for x in distances_sorted]


class HybridRecommenderClusterizzazione(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "HybridRecommenderClusterizzazione"

    def __init__(self, URM_train, ICM, recommender_list, tracks, XGBoost_model=None, UCM_train=None, dynamic=False,
                 d_weights=None, weights=None, XGB_model_ready=False,
                 URM_validation=None, sparse_weights=True, onPop=True, moreHybrids=False):
        super(Recommender, self).__init__()

        # CSR is faster during evaluation
        self.first_time = True
        self.pop = None
        self.UCM_train = UCM_train
        self.URM_train = check_matrix(URM_train, 'csr')
        self.URM_validation = URM_validation
        self.ICM = ICM
        self.dynamic = dynamic
        self.d_weights = d_weights
        self.dataset = None
        self.onPop = onPop
        self.moreHybrids = moreHybrids

        # parameters for xgboost
        self.user_id_XGBoost = None
        self.xgbModel = XGBoost_model
        self.xgb_model_ready = XGB_model_ready
        self.tracks = tracks
        self.UCM_dense = self.UCM_train.todense()
        self.ICM_dense = self.ICM.todense()

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

            else:  # UserCF, ItemCF, ItemCBF, P3alpha, RP3beta
                self.recommender_list.append(recommender(URM_train))

    def fit(self, topK=None, shrink=None, weights=None, pop=None, weights1=None, weights2=None, weights3=None,
            weights4=None,
            weights5=None, weights6=None, weights7=None, weights8=None, pop1=None, pop2=None, similarity='cosine',
            normalize=True,
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

    def change_weights(self, level, pop):
        if level < pop[0]:
            return self.d_weights[0]

        elif pop[0] < level < pop[1]:
            return self.d_weights[1]
        else:
            return self.d_weights[2]

    def getUserProfile(self, user_id):
        return self.URM_train.indices[
               self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

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

        cutoff_addition = 50
        cutoff_Boost = cutoff + cutoff_addition

        # compute the scores using the dot product
        # noinspection PyUnresolvedReferences

        if self.sparse_weights:
            scores = []
            # noinspection PyUnresolvedReferences
            for recommender in self.recommender_list:
                scores_batch = recommender.compute_item_score(user_id_array)

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
        ranking = []

        # i take the 20 elements with highest scores

        relevant_items_boost = (-final_score).argpartition(cutoff_Boost, axis=1)[:,
                               0:cutoff_Boost]
        songs_in_play = []
        final_relevant_items_boost = []
        for user_index in range(len(user_id_array)):
            user_id = user_id_array[user_index]
            old_rel = list(relevant_items_boost[user_index])
            profile_list = list(self.getUserProfile(user_id))
            to_append = old_rel + profile_list
            final_relevant_items_boost.append(to_append)

        relevant_items_boost = final_relevant_items_boost
        dict_song_pop = ged.tracks_popularity()

        # elements to add for each song

        user_list = user_id_array.tolist()

        tracks_duration_list = np.array(self.tracks['duration_sec']).reshape((-1, 1))[:, 0].tolist()

        for user_index in range(len(user_id_array)):
            user_relevant_items = relevant_items_boost[user_index]
            l = len(user_relevant_items)
            similarities_values = final_score[user_index, user_relevant_items].reshape((-1, 1))
            user_id = user_id_array[user_index]
            # Creating numpy array for training XGBoost

            song_pop = np.array([dict_song_pop[item] for item in user_relevant_items]).reshape((-1, 1))

            playlist_length = np.array([int(ged.lenght_playlist(self.getUserProfile(user_id)))] * l).reshape(
                (-1, 1))
            playlist_pop = np.array(
                [int(ged.playlist_popularity(self.getUserProfile(user_id), dict_song_pop))] * l).reshape(
                (-1, 1))

            # ucm_batch = self.UCM_train[user_list].toarray()
            ucm_batch = np.array([self.UCM_dense[user_id]] * l).reshape(l, -1)

            icm_batch = np.array([self.ICM_dense[item] for item in user_relevant_items]).reshape(l, -1)

            tracks_duration = np.array([tracks_duration_list[item] for item in user_relevant_items]).reshape((-1, 1))

            relevant_items_boost_user = np.asarray(user_relevant_items).reshape(-1, 1)
            newTrainXGBoost = np.concatenate(
                [relevant_items_boost_user, similarities_values, song_pop, playlist_pop, playlist_length,
                 tracks_duration, icm_batch, ucm_batch],
                axis=1)

            ranking.append(get_top_items(newTrainXGBoost, self.getUserProfile(user_id)))

        try:
            final_ranking = np.asarray(ranking, dtype=int)
        except ValueError:
            # Error here: ValueError: setting an array element with a sequence.
            print(ranking)
            print(user_id_array)
            raise ValueError

        assert final_ranking.shape[0] is len(user_id_array)
        assert final_ranking.shape[1] is 10
        return final_ranking
