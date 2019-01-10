#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/11/18

@author: Federico Betti
"""
import sys
import random

import time
import torch
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import DataLoader

from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import numpy as np
import sys, pickle

from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import Support_functions.get_evaluate_data as ged
from KNN.UserKNNCBFRecommender import UserKNNCBRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

import math


class HybridPytorch_SLIM(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "HybridPytorch"

    def __init__(self, URM_train, ICM, recommender_list, UCM_train=None, dynamic=False, d_weights=None, weights=None,
                 URM_validation=None, sparse_weights=True, onPop=True, batch_size=128, learning_rate=0.001):
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

        self.sparse_weights = sparse_weights

        self.recommender_list = []
        self.recommender_number = len(recommender_list)
        self.weights = weights

        self.normalize = None
        self.topK = None
        self.shrink = None

        # pytorch initialization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training = True

        use_cuda = False
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("MF_MSE_PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("MF_MSE_PyTorch: Using CPU")

        from MatrixFactorization.PyTorch.MF_MSE_PyTorch_model import MF_MSE_PyTorch_model, DatasetIterator_URM

        self.pyTorchModel = Hybridization_PyTorch_model(self.recommender_number).to(self.device)

        # Choose loss
        self.lossFunction = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.Adagrad(self.pyTorchModel.parameters(), lr=self.learning_rate)

        # Hybrid initialization
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
        tfidf_counter = 0

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

            elif recommender.__class__ in [ItemKNNCFRecommender]:
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim,
                                tfidf=similarity_args["tfidf"][tfidf_counter])
                tfidf_counter += 1

            else:  # ItemCF, UserCF, ItemCBF, UserCBF
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim)

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

                scores_batch = recommender.compute_item_score(user_id_array)
                # scores_batch = np.ravel(scores_batch) # because i'm not using batch

                if remove_top_pop_flag:
                    scores_batch = self._remove_TopPop_on_scores(scores_batch)

                if remove_CustomItems_flag:
                    scores_batch = self._remove_CustomItems_on_scores(scores_batch)

                scores.append(scores_batch)

            final_score = np.zeros(scores[0].shape)
            for user_index in range(len(user_id_array)):
                user_id = user_id_array[user_index]
                seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                seel_len = len(seen)

                target_var = np.ones((seel_len, 1))
                only_positive_samples = False
                if only_positive_samples:
                    negative_items = []
                else:
                    negative_items = random.sample(range(1, self.URM_train.shape[1]), seel_len)
                    negative_items = [x for x in negative_items if x not in list(seen)]
                    target_var_zeros = np.zeros((seel_len, 1))
                    target_var = np.concatenate((target_var, target_var_zeros))

                user_scores = np.zeros((scores_batch.shape[1], self.recommender_number))

                for rec_ind, score in enumerate(scores):
                    user_scores[:, rec_ind] = score[user_index].T

                # print("Seen: {}, negative items: {}".format(seen, negative_items))
                # print("Shapes {}, {}".format(user_scores[seen, :].shape, user_scores[negative_items, :].shape))
                if self.training:
                    input_var = np.concatenate((user_scores[seen, :], user_scores[negative_items, :]))

                    input_data_tensor = Variable(torch.Tensor(input_var)).to(self.device)

                    '''
                    label tensor with the WARP loss function should be a vector with 1 in the i-th position if the i-th
                    item is correctly recommended
                    '''
                    label_tensor = Variable(torch.Tensor(target_var)).to(self.device)

                    # FORWARD pass
                    prediction = self.pyTorchModel(input_data_tensor)
                    # print("Predictions on ones: {}".format(prediction))
                    # Pass prediction and label removing last empty dimension of prediction
                    loss = self.lossFunction(prediction.view(-1), label_tensor)

                    # BACKWARD pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                final_prediction = self.pyTorchModel(torch.Tensor(user_scores).to(self.device))
                # print(final_prediction)

                final_score[user_index] = final_prediction.detach().numpy().T
                final_score[user_index] = self._remove_seen_on_scores(user_id, final_score[user_index, :])


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

        # Creating numpy array for training XGBoost



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking


'''
This class is using pytorch models with WARP loss function
'''


class HybridPytorch_WARP(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "HybridPytorch_WARP"

    def __init__(self, URM_train, ICM, recommender_list, UCM_train=None, dynamic=False, d_weights=None, weights=None,
                 URM_validation=None, sparse_weights=True, onPop=True, batch_size=128, learning_rate=0.001):
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

        self.sparse_weights = sparse_weights

        self.recommender_list = []
        self.recommender_number = len(recommender_list)
        self.weights = weights

        self.normalize = None
        self.topK = None
        self.shrink = None

        # pytorch initialization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training = True

        use_cuda = False
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("MF_MSE_PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("MF_MSE_PyTorch: Using CPU")

        from MatrixFactorization.PyTorch.MF_MSE_PyTorch_model import MF_MSE_PyTorch_model, DatasetIterator_URM

        self.pyTorchModel = Hybridization_PyTorch_model(self.recommender_number).to(self.device)

        # Choose loss
        # self.lossFunction = torch.nn.MSELoss(size_average=False)
        self.lossFunction = WARPLoss()
        self.optimizer = torch.optim.Adagrad(self.pyTorchModel.parameters(), lr=self.learning_rate)

        # Hybrid initialization
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
        tfidf_counter = 0

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

            elif recommender.__class__ in [ItemKNNCFRecommender]:
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim,
                                tfidf=similarity_args["tfidf"][tfidf_counter])
                tfidf_counter += 1

            else:  # ItemCF, UserCF, ItemCBF, UserCBF
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim)

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
        remove_seen_flag = False or not self.training
        # compute the scores using the dot product
        # noinspection PyUnresolvedReferences
        if self.sparse_weights:
            scores = []
            # noinspection PyUnresolvedReferences
            for recommender in self.recommender_list:

                scores_batch = recommender.compute_item_score(user_id_array)
                # scores_batch = np.ravel(scores_batch) # because i'm not using batch

                if remove_seen_flag:
                    for user_index in range(len(user_id_array)):
                        user_id = user_id_array[user_index]

                        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

                        scores_batch[user_index, seen] = -100000

                if remove_top_pop_flag:
                    scores_batch = self._remove_TopPop_on_scores(scores_batch)

                if remove_CustomItems_flag:
                    scores_batch = self._remove_CustomItems_on_scores(scores_batch)

                scores.append(scores_batch)

            final_score = np.zeros(scores[0].shape)
            input_data_tensor, label_tensor = [], []

            print("Training users {}".format(user_id_array))
            for user_index in range(len(user_id_array)):
                user_id = user_id_array[user_index]

                test_songs = self.URM_validation.indices[
                             self.URM_validation.indptr[user_id]:self.URM_validation.indptr[user_id + 1]]
                # train_songs = self.URM_train.indices[
                #               self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                target_var = np.zeros((self.URM_train.shape[1], 1))
                target_var[test_songs] = 1
                # target_var[train_songs] = 1

                input_var = np.zeros((self.recommender_number, self.URM_train.shape[1]))
                for ind, score in enumerate(scores):
                    input_var[ind, :] = score[user_index]
                input_var = input_var.T.reshape(-1, 1)

                input_data_tensor.append(Variable(torch.Tensor(input_var)).to(self.device))
                label_tensor.append(Variable(torch.Tensor(target_var.reshape(1, -1))).to(self.device))

            input_data_tensor = torch.cat(input_data_tensor, dim=1)
            label_tensor = torch.cat(label_tensor, dim=0)

            if self.training:
                t = time.time()
                print("Start training on this batch")
                # FORWARD pass
                prediction = self.pyTorchModel(input_data_tensor)

                # print("Predictions on ones: {}".format(prediction))
                # Pass prediction and label removing last empty dimension of prediction
                loss = self.lossFunction(prediction.t(), label_tensor)

                # BACKWARD pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("Training loss: {} and last {} sec, {} sec per user".format(loss.data[0], time.time() - t, (time.time() - t) / len(user_id_array)))

            if self.training:
                return np.ones((len(user_id_array), cutoff), dtype=int)
            else:
                final_prediction = self.pyTorchModel(input_data_tensor)
                # print(final_prediction)
                final_score = final_prediction.detach().numpy().T

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

        # Creating numpy array for training XGBoost



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking


'''
The model with two hidden layers became slower and decrease the performances
'''


class WARP(Function):
    '''
    autograd function of WARP loss
    '''

    @staticmethod
    def forward(ctx, input, target, max_num_trials=None):

        batch_size = target.size()[0]
        if max_num_trials is None:
            max_num_trials = target.size()[1] - 1

        positive_indices = torch.zeros(input.size())
        negative_indices = torch.zeros(input.size())
        L = torch.zeros(input.size()[0])

        all_labels_idx = np.arange(target.size()[1])

        Y = float(target.size()[1])
        J = torch.nonzero(target)

        for i in range(batch_size):

            msk = np.ones(target.size()[1], dtype=bool)

            # Find the positive label for this example
            j = J[i, 1]
            positive_indices[i, j] = 1
            msk[j] = False

            # initialize the sample_score_margin
            sample_score_margin = -1
            num_trials = 0

            neg_labels_idx = all_labels_idx[msk]

            while ((sample_score_margin < 0) and (num_trials < max_num_trials)):
                # randomly sample a negative label
                neg_idx = random.sample(list(neg_labels_idx), 1)
                neg_idx = neg_idx[0]
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]

                num_trials += 1
                # calculate the score margin
                sample_score_margin = 1 + input[i, neg_idx] - input[i, j]

            if sample_score_margin < 0:
                # checks if no violating examples have been found
                continue
            else:
                loss_weight = np.log(math.floor((Y - 1) / (num_trials)))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1

        loss = L * (1 - torch.sum(positive_indices * input, dim=1) + torch.sum(negative_indices * input, dim=1))
        #
        # ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices

        return torch.sum(loss, dim=0, keepdim=True)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # input, target = ctx.saved_variables
        L = Variable(torch.unsqueeze(ctx.L, 1), requires_grad=False)

        positive_indices = Variable(ctx.positive_indices, requires_grad=False)
        negative_indices = Variable(ctx.negative_indices, requires_grad=False)
        grad_input = grad_output * L * (negative_indices - positive_indices)

        return grad_input, None, None


class WARPLoss(torch.nn.Module):
    def __init__(self, max_num_trials=None):
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials

    def forward(self, input, target):
        return WARP.apply(input, target, self.max_num_trials)


class tiedLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super(tiedLinear, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        print("input shape {}".format(input.size()))
        print(self.weight, self.weight.size())
        repeated_weight = self.weight.repeat(input.size()[0], int(input.size()[0] / self.in_features))
        print(repeated_weight, repeated_weight.size())
        return F.linear(input, repeated_weight, self.bias)


class Hybridization_PyTorch_model(torch.nn.Module):
    def __init__(self, n_recommenders):
        super(Hybridization_PyTorch_model, self).__init__()

        self.n_recommenders = n_recommenders

        self.layer_1 = torch.nn.Linear(in_features=self.n_recommenders, out_features=1, bias=False)
        self.layer_1.weight.data.fill_(1)
        # self.activation_function1 = torch.nn.ReLU()
        self.activation_function2 = torch.nn.Sigmoid()

    def forward_old(self, input):
        # if I suppose that the input is [ 8 * 10k]
        # for i in range(0, input.shape[0], self.n_recommenders):
        #     item_input =
        # print("Input: {}".format(input))
        prediction = self.layer_1(input)
        # prediction = self.activation_function1(prediction)
        # prediction = self.layer_2(prediction)
        # print("Prediction1: {}".format(prediction))
        # prediction = self.activation_function2(prediction)
        # print("Prediction final: {}".format(prediction))

        return prediction

    def forward(self, input):
        total_predictions, predictions = [], []
        for user in range(input.size()[1]):
            predictions = []
            user_input = input.narrow(1, user, 1).squeeze()

            for i in range(0, input.shape[0], self.n_recommenders):
                predictions.append(self.layer_1(user_input.narrow(0, i, self.n_recommenders)))

            user_final_proability = torch.cat(predictions, 0).view(-1, 1)
            total_predictions.append(torch.div(user_final_proability, user_final_proability.max()))

        prediction = torch.cat(total_predictions, 1)
        print("Linear model weights: {} and bias: {}".format(self.layer_1.weight, self.layer_1.bias))
        return prediction
