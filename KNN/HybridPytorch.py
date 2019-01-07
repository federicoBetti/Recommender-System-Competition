#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/11/18

@author: Federico Betti
"""
import sys
import random

import torch
from torch.autograd import Variable
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


class HybridPytorch(SimilarityMatrixRecommender, Recommender):
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
        # self.lossFunction = torch.nn.BCELoss(size_average=False)
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
The model with two hidden layers became slower and decrease the performances
'''
class Hybridization_PyTorch_model(torch.nn.Module):
    def __init__(self, n_recommenders):
        super(Hybridization_PyTorch_model, self).__init__()

        self.n_recommenders = n_recommenders

        self.layer_1 = torch.nn.Linear(in_features=self.n_recommenders, out_features=32)
        self.layer_2 = torch.nn.Linear(in_features=32, out_features=1)

        self.activation_function1 = torch.nn.ReLU()
        self.activation_function2 = torch.nn.Sigmoid()

    def forward(self, input):
        # print("Input: {}".format(input))
        prediction = self.layer_1(input)
        prediction = self.activation_function1(prediction)
        prediction = self.layer_2(prediction)
        # print("Prediction1: {}".format(prediction))
        prediction = self.activation_function2(prediction)
        # print("Prediction final: {}".format(prediction))

        return prediction


'''
class Hybridization_PyTorch_model(Recommender, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "MF_MSE_PyTorch_Recommender"

    def __init__(self, URM_train, positive_threshold=1, URM_validation=None):

        super(Hybridization_PyTorch_model, self).__init__()

        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False

        self.positive_threshold = positive_threshold

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        self.compute_item_score = self.compute_score_MF

    def compute_score_MF(self, user_id):

        scores_array = np.dot(self.W[user_id], self.H.T)

        return scores_array

    def fit(self, epochs=30, batch_size=1024, num_factors=10,
            learning_rate=0.001,
            stop_on_validation=False, lower_validatons_allowed=5, validation_metric="MAP",
            evaluator_object=None, validation_every_n=1, use_cuda=True):

        if evaluator_object is None and self.URM_validation is not None:
            from Base.Evaluation.Evaluator import SequentialEvaluator

            evaluator_object = SequentialEvaluator(self.URM_validation, [10])

        self.n_factors = num_factors

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
        URM_train_positive.eliminate_zeros()

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        ########################################################################################################
        #
        #                                SETUP PYTORCH MODEL AND DATA READER
        #
        ########################################################################################################

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("MF_MSE_PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("MF_MSE_PyTorch: Using CPU")

        from MatrixFactorization.PyTorch.MF_MSE_PyTorch_model import MF_MSE_PyTorch_model, DatasetIterator_URM

        n_users, n_items = self.URM_train.shape

        self.pyTorchModel = MF_MSE_PyTorch_model(n_users, n_items, self.n_factors).to(self.device)

        # Choose loss
        self.lossFunction = torch.nn.MSELoss(size_average=False)
        # self.lossFunction = torch.nn.BCELoss(size_average=False)
        self.optimizer = torch.optim.Adagrad(self.pyTorchModel.parameters(), lr=self.learning_rate)

        dataset_iterator = DatasetIterator_URM(self.URM_train)

        self.train_data_loader = DataLoader(dataset=dataset_iterator,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            # num_workers = 2,
                                            )

        ########################################################################################################


        self._train_with_early_stopping(epochs, validation_every_n, stop_on_validation,
                                        validation_metric, lower_validatons_allowed, evaluator_object,
                                        algorithm_name="MF_MSE_PyTorch")

        self.W = self.W_best.copy()
        self.H = self.H_best.copy()

        sys.stdout.flush()

    def _initialize_incremental_model(self):

        self.W_incremental = self.pyTorchModel.get_W()
        self.W_best = self.W_incremental.copy()

        self.H_incremental = self.pyTorchModel.get_H()
        self.H_best = self.H_incremental.copy()

    def _update_incremental_model(self):

        self.W_incremental = self.pyTorchModel.get_W()
        self.H_incremental = self.pyTorchModel.get_H()

        self.W = self.W_incremental.copy()
        self.H = self.H_incremental.copy()

    def _update_best_model(self):

        self.W_best = self.W_incremental.copy()
        self.H_best = self.H_incremental.copy()

    def _run_epoch(self, num_epoch):

        for num_batch, (input_data, label) in enumerate(self.train_data_loader, 0):

            if num_batch % 1000 == 0:
                print("num_batch: {}".format(num_batch))

            print("Batch number {} with input data shape: {}".format(num_batch, input_data.shape))
            # On windows requires int64, on ubuntu int32
            # input_data_tensor = Variable(torch.from_numpy(np.asarray(input_data, dtype=np.int64))).to(self.device)
            input_data_tensor = Variable(input_data).to(self.device)

            label_tensor = Variable(label).to(self.device)

            user_coordinates = input_data_tensor[:, 0]
            item_coordinates = input_data_tensor[:, 1]

            # FORWARD pass
            prediction = self.pyTorchModel(user_coordinates, item_coordinates)

            # Pass prediction and label removing last empty dimension of prediction
            loss = self.lossFunction(prediction.view(-1), label_tensor)

            # BACKWARD pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'learn_rate': self.learning_rate,
                          'num_factors': self.n_factors,
                          'batch_size': 1,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            logFile.flush()

    def saveModel(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"W": self.W,
                              "H": self.H}

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        np.savez(folder_path + "{}.npz".format(file_name), W=self.W, H=self.H)
        '''
