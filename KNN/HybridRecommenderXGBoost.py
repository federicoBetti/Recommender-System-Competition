#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from abc import ABC

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import numpy as np
from Dataset.RS_Data_Loader import RS_Data_Loader
from scipy import sparse
import xgboost as xgb
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFPageRankRecommender import ItemKNNCFPageRankRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import Support_functions.get_evaluate_data as ged
from KNN.UserKNNCBFRecommender import UserKNNCBRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import bisect


class HybridRecommenderXGBoost(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "HybridRecommenderXGBoost"

    def __init__(self, URM_train, ICM, recommender_list, URM_PageRank_train=None,
                 XGBoost_model=None, UCM_train=None, dynamic=False,
                 d_weights=None, weights=None, XGB_model_ready=False,
                 URM_validation=None, sparse_weights=True, onPop=True, moreHybrids=False, tracks=None):
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
        self.URM_PageRank_train = URM_PageRank_train

        # parameters for xgboost
        self.user_id_XGBoost = None
        self.xgbModel = XGBoost_model
        self.xgb_model_ready = XGB_model_ready
        self.tracks = tracks
        self.UCM_dense = self.UCM_train.todense()
        self.ICM_dense = self.ICM.todense()

        self.dict_song_pop = ged.tracks_popularity()
        self.tracks_duration_list = np.array(self.tracks['duration_sec']).reshape((-1, 1))[:, 0].tolist()
        self.all_tracks = range(0, URM_train.shape[1])

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
            elif recommender.__class__ in [HybridRecommenderXGBoost]:
                self.recommender_list.append(recommender)
            elif recommender in [ItemKNNCFPageRankRecommender]:
                self.recommender_list.append(recommender(self.URM_train, self.URM_PageRank_train))

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

    def xgboost_data_prediction(self, user_id_array, relevant_items_boost, cutoff_Boost):
        dict_song_pop = ged.tracks_popularity()

        # elements to add for each song

        user_list = user_id_array.tolist()

        tracks_duration_list = np.array(self.tracks['duration_sec']).reshape((-1, 1))[:, 0].tolist()

        song_pop = np.array([[dict_song_pop[item] for item in relevant_line]
                             for relevant_line in relevant_items_boost.tolist()]).reshape((-1, 1))

        playlist_length = np.array([[int(ged.lenght_playlist(self.getUserProfile(user)))] * cutoff_Boost
                                    for user in user_list]).reshape((-1, 1))
        playlist_pop = np.array([[int(ged.playlist_popularity(self.getUserProfile(user), dict_song_pop))] * cutoff_Boost
                                 for user in user_list]).reshape((-1, 1))

        '''
        # ucm_batch = self.UCM_train[user_list].toarray()
        dim_ucm = int(len(user_list) * 20)
        ucm_batch = np.array([self.UCM_dense[user] for _ in range(cutoff_Boost)
                              for user in user_list]).reshape(dim_ucm, -1)
       
        dim_icm = int(len(relevant_items_boost) * 20)
        icm_batch = np.array([[self.ICM_dense[item] for item in relevant_line]
                              for relevant_line in relevant_items_boost.tolist()], dtype=int).reshape(dim_icm, -1)
        '''
        tracks_duration = np.array([[tracks_duration_list[item] for item in relevant_line]
                                    for relevant_line in relevant_items_boost.tolist()]).reshape((-1, 1))

        relevant_items_boost = relevant_items_boost.reshape(-1, 1)
        return np.concatenate([relevant_items_boost, song_pop, playlist_pop, playlist_length,
                               tracks_duration], axis=1)  #icm_batch , ucm_batch],

    def mean_pl_length(self, URM, songs):
        length_list = []
        for song in songs:
            length = 0
            relevant_users = np.argwhere(URM[:, song])[:, 0]
            for user in relevant_users:
                length += int(ged.lenght_playlist(self.getUserProfile(user)))

            if relevant_users.shape[0] == 0:
                length_list.append(0)
                continue
            length = int(length / relevant_users.shape[0])
            length_list.append(length)

        return length_list

    def mean_pl_pop(self, URM, songs, pop_dict):
        pop_list = []
        for song in songs:
            pop = 0
            relevant_users = np.argwhere(URM[:, song])[:, 0]
            for user in relevant_users:
                pop += int(ged.playlist_popularity(self.getUserProfile(user), pop_dict))

            if relevant_users.shape[0] == 0:
                pop_list.append(0)
                continue

            pop = int(pop / relevant_users.shape[0])
            pop_list.append(pop)

        return pop_list

    def xgboost_data_training(self, user_id, URM_train):
        # elements to add for each song

        playlist_pos_song_cut = URM_train[user_id].indices.tolist()
        playlist_neg_song_cut = random.sample([x for x in self.all_tracks if x not in playlist_pos_song_cut],
                                              len(playlist_pos_song_cut))
        playlist_songs_selected = playlist_pos_song_cut + playlist_neg_song_cut
        playlist_songs_selected_array = np.array(playlist_songs_selected).reshape(-1, 1)

        songs_pop = np.array([self.dict_song_pop[item] for item in playlist_songs_selected]).reshape(-1, 1)
        mean_playlist_length = np.array(self.mean_pl_length(URM_train, playlist_songs_selected)).reshape(-1, 1)
        mean_playlist__pop = np.array(self.mean_pl_pop(URM_train, playlist_songs_selected,
                                                       self.dict_song_pop)).reshape(-1, 1)
        tracks_duration = np.array([self.tracks_duration_list[item] for item in playlist_songs_selected]).reshape(-1, 1)

        '''
        # ucm_batch = self.UCM_train[user_list].toarray()
        dim_ucm = int(len(user_list) * 20)
        ucm_batch = np.array([self.UCM_dense[user] for _ in range(cutoff_Boost)
                              for user in user_list]).reshape(dim_ucm, -1)
      
        icm_batch = np.array([self.ICM_dense[item] for item in
                              playlist_songs_selected], dtype=int).reshape(len(playlist_songs_selected), -1)
        '''
        return np.concatenate([playlist_songs_selected_array, songs_pop, mean_playlist__pop, mean_playlist_length,
                               tracks_duration], axis=1)  #icm_batch , ucm_batch],

    def reorder_songs(self, preds, user_recommendations):

        ordered_tracks = []
        count = 0
        for track in user_recommendations[:, 0]:
            ordered_tracks.append((track, (preds[count][0])))
            count += 1

        ordered_tracks.sort(key=lambda elem: elem[1])
        return [track_id[0] for track_id in ordered_tracks]

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

        cutoff_addition = 10
        cutoff_Boost = cutoff + cutoff_addition

        # compute the scores using the dot product
        # noinspection PyUnresolvedReferences

        if self.sparse_weights:
            scores = []
            # noinspection PyUnresolvedReferences
            for recommender in self.recommender_list:
                if recommender.__class__ in [HybridRecommenderXGBoost]:
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
                    # weights = self.change_weights(user_id)
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

        # i take the 20 elements with highest scores

        relevant_items_boost = (-final_score).argpartition(cutoff_Boost, axis=1)[:,
                               0:cutoff_Boost]

        # if not self.xgb_model_ready:
        relevant_items_partition = (-final_score).argpartition(cutoff, axis=1)[:, 0:cutoff]

        relevant_items_partition_original_value = final_score[
            np.arange(final_score.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        # Creating numpy array for training XGBoost
        data_reader = RS_Data_Loader()
        URM_train = data_reader.get_URM_train()

        pred_data_xgboost = self.xgboost_data_prediction(user_id_array, relevant_items_boost, cutoff_Boost)

        param = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': 2}  # the number of classes that exist in this datset
        num_round = 20

        ranking = []
        for user_index in range(len(user_id_array)):
            user_id = user_id_array[user_index]
            # if self.user_id_XGBoost is None:
            #     self.user_id_XGBoost = np.array([user_id] * cutoff_Boost).reshape(-1, 1)
            # else:
            #     self.user_id_XGBoost = np.concatenate([self.user_id_XGBoost,
            #                                            np.array([user_id] *
            #                                                     cutoff_Boost).reshape(-1, 1)], axis=0)

            train_xgboost = self.xgboost_data_training(user_id, URM_train)
            half_play = int(train_xgboost.shape[0] / 2)
            labels_train = np.array([1] * half_play + [0] * half_play)
            dtrain = xgb.DMatrix(train_xgboost, label=labels_train)
            bst = xgb.train(param, dtrain, num_round)

            user_recommendations = pred_data_xgboost[user_index * cutoff_Boost:(user_index + 1) * cutoff_Boost]
            dtest = xgb.DMatrix(user_recommendations)
            preds = bst.predict(dtest)
            predictions = self.reorder_songs(preds, user_recommendations)[:cutoff]
            ranking.append(predictions)
            print(user_id, predictions)

        #
        # if self.xgb_model_ready:
        #     print("QUI")
        #     preds = self.xgbModel.predict_proba(newTrainXGBoost)
        #     # preds = self.xgbModel.predict(newTrainXGBoost)
        #     ranking = []
        #     ordered_tracks = []
        #     current_user_id = 0
        #     current_user = user_list[current_user_id]
        #     for track_idx in range(newTrainXGBoost.shape[0]):
        #         ordered_tracks.append((relevant_items_boost[track_idx], preds[track_idx][current_user]))
        #
        #         if track_idx % cutoff_Boost and track_idx != 0:
        #             ordered_tracks.sort(key=lambda elem: elem[1])
        #             ordered_tracks = [track_id[0] for track_id in ordered_tracks]
        #             ranking.append(ordered_tracks)
        #             ordered_tracks = []
        #             current_user_id += 1
        #
        #
        # elif not self.xgb_model_ready:
        #     if self.first_time:
        #         self.first_time = False
        #         self.trainXGBoost = sparse.lil_matrix(newTrainXGBoost, dtype=int)
        #         x = self.trainXGBoost
        #         y = self.user_id_XGBoost
        #         print()
        #
        #     elif not self.first_time:
        #         self.trainXGBoost = sparse.vstack([self.trainXGBoost, newTrainXGBoost], dtype=int)
        #         x = self.trainXGBoost
        #         y = 0
        # Return single list for one user, instead of list of lists
        # if single_user:
        #     ranking_list = ranking_list[0]

        return ranking

    '''
    QUESTA L'HO SOLO COMMENTATA PER CREARNE UNA SOPRA, QUESTA È QUELLA VECCHIA CHE AVEVAMO FATTO PRIMA
    '''

    # def recommend_OLD(self, user_id_array, dict_pop=None, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
    #                   remove_CustomItems_flag=False):
    #
    #     if np.isscalar(user_id_array):
    #         user_id_array = np.atleast_1d(user_id_array)
    #         single_user = True
    #     else:
    #         single_user = False
    #
    #     weights = self.weights
    #     if cutoff == None:
    #         # noinspection PyUnresolvedReferences
    #         cutoff = self.URM_train.shape[1] - 1
    #     else:
    #         cutoff
    #
    #     cutoff_addition = 10
    #     cutoff_Boost = cutoff + cutoff_addition
    #
    #     # compute the scores using the dot product
    #     # noinspection PyUnresolvedReferences
    #
    #     if self.sparse_weights:
    #         scores = []
    #         # noinspection PyUnresolvedReferences
    #         for recommender in self.recommender_list:
    #             if recommender.__class__ in [HybridRecommenderXGBoost]:
    #                 scores.append(self.compute_score_hybrid(recommender, user_id_array, dict_pop,
    #                                                         remove_seen_flag=True, remove_top_pop_flag=False,
    #                                                         remove_CustomItems_flag=False))
    #
    #                 continue
    #             scores_batch = recommender.compute_item_score(user_id_array)
    #             # scores_batch = np.ravel(scores_batch) # because i'm not using batch
    #
    #             for user_index in range(len(user_id_array)):
    #
    #                 user_id = user_id_array[user_index]
    #
    #                 if remove_seen_flag:
    #                     scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])
    #
    #             if remove_top_pop_flag:
    #                 scores_batch = self._remove_TopPop_on_scores(scores_batch)
    #
    #             if remove_CustomItems_flag:
    #                 scores_batch = self._remove_CustomItems_on_scores(scores_batch)
    #
    #             scores.append(scores_batch)
    #
    #         final_score = np.zeros(scores[0].shape)
    #
    #         if self.dynamic:
    #             for user_index in range(len(user_id_array)):
    #                 user_id = user_id_array[user_index]
    #                 user_profile = self.URM_train.indices[
    #                                self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
    #                 if self.onPop:
    #                     level = int(ged.playlist_popularity(user_profile, dict_pop))
    #                 else:
    #                     level = int(ged.lenght_playlist(user_profile))
    #                 # weights = self.change_weights(user_id)
    #                 weights = self.change_weights(level, self.pop)
    #                 assert len(weights) == len(scores), "Scores and weights have different lengths"
    #
    #                 final_score_line = np.zeros(scores[0].shape[1])
    #                 if sum(weights) > 0:
    #                     for score, weight in zip(scores, weights):
    #                         final_score_line += score[user_index] * weight
    #                 final_score[user_index] = final_score_line
    #         else:
    #             for score, weight in zip(scores, weights):
    #                 final_score += (score * weight)
    #
    #     else:
    #         raise NotImplementedError
    #
    #     # i take the 20 elements with highest scores
    #
    #     for user_index in range(len(user_id_array)):
    #
    #         user_id = user_id_array[user_index]
    #         if self.user_id_XGBoost is None:
    #             self.user_id_XGBoost = np.array([user_id] * cutoff_Boost).reshape(-1, 1)
    #         else:
    #             self.user_id_XGBoost = np.concatenate([self.user_id_XGBoost,
    #                                                    np.array([user_id] *
    #                                                             cutoff_Boost).reshape(-1, 1)], axis=0)
    #
    #     relevant_items_boost = (-final_score).argpartition(cutoff_Boost, axis=1)[:,
    #                            0:cutoff_Boost]
    #
    #     if not self.xgb_model_ready:
    #         relevant_items_partition = (-final_score).argpartition(cutoff, axis=1)[:, 0:cutoff]
    #
    #         relevant_items_partition_original_value = final_score[
    #             np.arange(final_score.shape[0])[:, None], relevant_items_partition]
    #         relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
    #         ranking = relevant_items_partition[
    #             np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]
    #
    #     # Creating numpy array for training XGBoost
    #
    #     dict_song_pop = ged.tracks_popularity()
    #
    #     # elements to add for each song
    #
    #     user_list = user_id_array.tolist()
    #
    #     tracks_duration_list = np.array(self.tracks['duration_sec']).reshape((-1, 1))[:, 0].tolist()
    #
    #     song_pop = np.array([[dict_song_pop[item] for item in relevant_line]
    #                          for relevant_line in relevant_items_boost.tolist()]).reshape((-1, 1))
    #
    #     playlist_length = np.array([[int(ged.lenght_playlist(self.getUserProfile(user)))] * cutoff_Boost
    #                                 for user in user_list]).reshape((-1, 1))
    #     playlist_pop = np.array([[int(ged.playlist_popularity(self.getUserProfile(user), dict_song_pop))] * cutoff_Boost
    #                              for user in user_list]).reshape((-1, 1))
    #
    #     '''
    #     # ucm_batch = self.UCM_train[user_list].toarray()
    #     dim_ucm = int(len(user_list) * 20)
    #     ucm_batch = np.array([self.UCM_dense[user] for _ in range(cutoff_Boost)
    #                           for user in user_list]).reshape(dim_ucm, -1)
    #
    #     dim_icm = int(len(relevant_items_boost) * 20)
    #     icm_batch = np.array([[self.ICM_dense[item] for item in relevant_line]
    #                           for relevant_line in relevant_items_boost.tolist()]).reshape(dim_icm, -1)
    #     '''
    #     tracks_duration = np.array([[tracks_duration_list[item] for item in relevant_line]
    #                                 for relevant_line in relevant_items_boost.tolist()]).reshape((-1, 1))
    #
    #     relevant_items_boost = relevant_items_boost.reshape(-1, 1)
    #     newTrainXGBoost = np.concatenate([relevant_items_boost, song_pop, playlist_pop, playlist_length,
    #                                       tracks_duration],  # icm_batch, ucm_batch],
    #                                      axis=1)
    #
    #     if self.xgb_model_ready:
    #         print("QUI")
    #         preds = self.xgbModel.predict(newTrainXGBoost)
    #         ranking = []
    #         ordered_tracks = []
    #         current_user_id = 0
    #         current_user = user_list[current_user_id]
    #         for track_idx in range(newTrainXGBoost.shape[0]):
    #             ordered_tracks.append((relevant_items_boost[track_idx], preds[track_idx][current_user]))
    #
    #             if track_idx % cutoff_Boost and track_idx != 0:
    #                 ordered_tracks.sort(key=lambda elem: elem[1])
    #                 ordered_tracks = [track_id[0] for track_id in ordered_tracks]
    #                 ranking.append(ordered_tracks)
    #                 ordered_tracks = []
    #                 current_user_id += 1
    #
    #
    #     elif not self.xgb_model_ready:
    #         if self.first_time:
    #             self.first_time = False
    #             self.trainXGBoost = sparse.lil_matrix(newTrainXGBoost, dtype=int)
    #
    #         elif not self.first_time:
    #             self.trainXGBoost = sparse.vstack([self.trainXGBoost, newTrainXGBoost], dtype=int)
    #             x = self.trainXGBoost
    #             y = 0
    #     # Return single list for one user, instead of list of lists
    #     # if single_user:
    #     #     ranking_list = ranking_list[0]
    #
    #     return ranking
