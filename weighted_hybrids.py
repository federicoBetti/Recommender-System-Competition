import sys

import time

from Dataset.RS_Data_Loader import RS_Data_Loader
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.HybridRecommenderTopNapproach import HybridRecommenderTopNapproach
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.HybridRecommender import HybridRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBRecommender
import Support_functions.get_evaluate_data as ged
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender

from data.Movielens_10M.Movielens10MReader import Movielens10MReader

import traceback, os

import Support_functions.manage_data as md
from run_parameter_search import delete_previous_intermediate_computations
import numpy as np
from scipy import sparse as sps


def get_user_relevant_items(user_id, URM_test):
    return URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]


def map(is_relevant, pos_items):
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([pos_items.shape[0], is_relevant.shape[0]])

    assert 0 <= map_score <= 1, map_score
    return map_score


if __name__ == '__main__':
    # delete_previous_intermediate_computations()
    dataReader = RS_Data_Loader()

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    UCM_tfidf = dataReader.get_tfidf_artists()
    # _ = dataReader.get_tfidf_album()

    recommender_list = [
        # Random,
        # TopPop,
        ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, URM_train, exclude_seen=True)

    logFile = open("result_experiments/" + "result_weighted_hybrids.txt", "a")
    dict_song_pop = ged.tracks_popularity()

    URM_test_list = URM_test
    if not isinstance(URM_test_list, list):
        URM_test = URM_test_list.copy()
        URM_test_list = [URM_test_list]
    else:
        raise ValueError("List of URM_test not supported")

    n_users = URM_test_list[0].shape[0]
    n_items = URM_test_list[0].shape[1]

    # Prune users with an insufficient number of ratings
    # During testing CSR is faster
    URM_test_list2 = []
    usersToEvaluate_mask = np.zeros(n_users, dtype=np.bool)

    for URM_test in URM_test_list:
        URM_test = sps.csr_matrix(URM_test)
        URM_test_list2.append(URM_test)

        rows = URM_test.indptr
        numRatings = np.ediff1d(rows)
        new_mask = numRatings >= 1

        usersToEvaluate_mask = np.logical_or(usersToEvaluate_mask, new_mask)

    usersToEvaluate = np.arange(n_users)[usersToEvaluate_mask]
    usersToEvaluate = list(usersToEvaluate)

    try:
        recommender_class = HybridRecommender
        print("Algorithm: {}".format(recommender_class))

        onPop = True

        lambda_i = 0.1
        lambda_j = 0.05
        old_similrity_matrix = None
        num_factors = 165
        l1_ratio = 1e-06
        # alpha = [1.9959734038074426, 0.10609858937191907, 0.4608371142966865, 1.905978868103585, 1.6329874929834254, 1.7878599729785276]
        # alpha = [1 / len(recommender_list)]*len(recommender_list)
        alpha = [1 / 3, 1 / 3, 1 / 3]
        i = 0
        while i < 80:
            recommender_class = HybridRecommender
            recommender = recommender_class(URM_train, ICM, recommender_list, UCM_train=UCM_tfidf, dynamic=False,
                                            # d_weights=d_weights,
                                            URM_validation=URM_validation, onPop=onPop)
            MAP = 0
            print("alpha: {}".format(alpha))
            recommender.fit(**{"topK": [10, 140, 160],
                               "shrink": [180, 1, 2],
                               "pop": [130, 346],
                               "weights": alpha,
                               "force_compute_sim": False,
                               "old_similarity_matrix": old_similrity_matrix,
                               "epochs": 50, "lambda_i": lambda_i,
                               "lambda_j": lambda_j,
                               "num_factors": num_factors,
                               'alphaP3': 1.160296393373262,
                               'alphaRP3': 0.4156476217553893,
                               'betaRP': 0.20430089442930188,
                               'l1_ratio': l1_ratio,
                               "weights_to_dweights": 1})

            print("Starting Evaluations...")
            t = time.time()
            user_batch_start = 0
            user_batch_end = 0
            n_users_evaluated = 0

            start_time = time.time()
            start_time_print = time.time()
            while user_batch_start < len(usersToEvaluate):
                user_batch_end = user_batch_start + 1000
                user_batch_end = min(user_batch_end, len(usersToEvaluate))

                test_user_batch_array = np.array(usersToEvaluate[user_batch_start:user_batch_end])
                user_batch_start = user_batch_end

                # Compute predictions for a batch of users using vectorization, much more efficient than computing it one
                # at a time
                recommended_items_batch_list = recommender.recommend_gradient_descent(test_user_batch_array,
                                                                                      remove_seen_flag=True,
                                                                                      cutoff=10,
                                                                                      remove_top_pop_flag=False,
                                                                                      remove_CustomItems_flag=False,
                                                                                      dict_pop=dict_song_pop)
                # Compute recommendation quality for each user in batch
                for batch_user_index in range(len(recommended_items_batch_list)):
                    n_users_evaluated += 1
                    user_id = test_user_batch_array[batch_user_index]
                    recommended_items = recommended_items_batch_list[batch_user_index]

                    # Being the URM CSR, the indices are the non-zero column indexes
                    relevant_items = get_user_relevant_items(user_id, URM_test)
                    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                    user_profile = URM_train.indices[
                                   URM_train.indptr[user_id]:URM_train.indptr[user_id + 1]]
                    key_pop = int(ged.playlist_popularity(user_profile, pop_dict=dict_song_pop))
                    key_len = int(ged.lenght_playlist(user_profile))

                    for cutoff in [10]:
                        is_relevant_current_cutoff = is_relevant[0:cutoff]
                        recommended_items_current_cutoff = recommended_items[0:cutoff]

                        current_map = map(is_relevant_current_cutoff, relevant_items)
                        MAP += current_map

                    if time.time() - start_time_print > 30 or n_users_evaluated == len(usersToEvaluate):
                        print(
                            "SequentialEvaluator: Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                n_users_evaluated,
                                100.0 * float(n_users_evaluated) / len(usersToEvaluate),
                                time.time() - start_time,
                                float(n_users_evaluated) / (time.time() - start_time)))
                        start_time_print = time.time()

            H = len(URM_test.data)
            print("Recommendations done in {}, with user_per_second = {}".format(time.time() - t,
                                                                                 URM_test.shape[0] / (time.time() - t)))
            grad = [x / H for x in recommender.gradients]
            MAE = recommender.MAE / H
            print("MAE: {}, MAP: {},  gradients: {}".format(MAE, MAP / n_users_evaluated, grad))
            # learning rate
            lr = 0.1
            # gradient descent
            alpha = [a - lr * g for a, g in zip(alpha, grad)]
            # normalization
            alpha = [float(index) / sum(alpha) for index in alpha]

            logFile.write(
                "Iteration: {}, MAE: {}, MAP: {},  gradients: {}\n".format(i, MAE, MAP / n_users_evaluated, grad))
            logFile.flush()
            i += 1

    except Exception as e:
        traceback.print_exc()
        logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
        logFile.flush()
