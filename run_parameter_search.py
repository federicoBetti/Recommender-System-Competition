#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
import glob

from Base.Evaluation.Evaluator import SequentialEvaluator
from Base.NonPersonalizedRecommender import TopPop, Random
from Dataset.RS_Data_Loader import RS_Data_Loader
from KNN.HybridRecommender import HybridRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from ParameterTuning.RandomSearch import RandomSearch
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from Support_functions import get_evaluate_data as ged

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from ParameterTuning.BayesianSearch import BayesianSearch

import traceback, pickle
from Utils.PoolWithSubprocess import PoolWithSubprocess

from ParameterTuning.AbstractClassSearch import DictionaryKeys
import numpy as np

def run_KNNCFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, n_cases, output_root_path,
                                            metric_to_optimize):
    # pay attention that it doesn't finish (it should after n_cases, but now it doens't work)
    # it saves best results in the txt file
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),  # questi sono quelli fissi
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}  # questi sono quelli che stai testando

    output_root_path_similarity = output_root_path + "_" + similarity_type

    # questo runna fit del recommender
    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize)
    print(best_parameters)


def run_HybridRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, ICM, n_cases,
                                             output_root_path,
                                             metric_to_optimize):
    # pay attention that it doesn't finish (it should after n_cases, but now it doens't work)
    # it saves best results in the txt file
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = output_root_path + "_" + similarity_type

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize)


def run_KNNCBFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, ICM_train, n_cases,
                                             output_root_path, metric_to_optimize):
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    if similarity_type in ["cosine", "asymmetric"]:
        hyperparamethers_range_dictionary["feature_weighting"] = ["none", "BM25", "TF-IDF"]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [ICM_train, URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = output_root_path + "_" + similarity_type

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize)


def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name, n_cases=30,
                               evaluator_validation=None, evaluator_test=None, metric_to_optimize="PRECISION",
                               output_root_path="result_experiments/", parallelizeKNN=False):
    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    ##########################################################################################################

    this_output_root_path = output_root_path + recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation,
                                     evaluator_test=evaluator_test)

    similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNCBFRecommender_on_similarity_type,
                                                               parameterSearch=parameterSearch,
                                                               URM_train=URM_train,
                                                               ICM_train=ICM_object,
                                                               n_cases=n_cases,
                                                               output_root_path=this_output_root_path,
                                                               metric_to_optimize=metric_to_optimize)

    if parallelizeKNN:
        pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)


def runParameterSearch_Hybrid_partial(recommender_class, URM_train, ICM, recommender_list, n_cases=35,
                                      evaluator_validation=None, evaluator_test=None, metric_to_optimize="MAP",
                                      output_root_path="result_experiments/", parallelizeKNN=False, URM_test=None,
                                      old_similrity_matrix=None):
    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    ##########################################################################################################

    this_output_root_path = output_root_path + "Hybrid:" + "{}".format([x.RECOMMENDER_NAME for x in recommender_list])

    # since test and validation are the same for now, here I don't pass the evaluator test (otherwise it also crash)
    parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation)

    similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    hyperparamethers_range_dictionary = {}
    '''
    Those are not parameters to test so the can be passed directly as FIT_KEYWORD_ARGS
    hyperparamethers_range_dictionary["topK1"] = [60]
    hyperparamethers_range_dictionary["topK2"] = [200]
    hyperparamethers_range_dictionary["topK3"] = [200]
    hyperparamethers_range_dictionary["shrink1"] = [5]
    hyperparamethers_range_dictionary["shrink2"] = [15]
    hyperparamethers_range_dictionary["shrink3"] = [5]
    I left all the params in the hybrid fit function just in case we want to use those again
    '''

    hyperparamethers_range_dictionary["weights1"] = range(0, 1)
    hyperparamethers_range_dictionary["weights2"] = range(0.5, 1)
    hyperparamethers_range_dictionary["weights3"] = range(0, 1)
    hyperparamethers_range_dictionary["weights4"] = range(0, 1)
    # hyperparamethers_range_dictionary["weights5"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    lambda_i = 0.01
    lambda_j = 0.01
    old_similrity_matrix = None
    num_factors = 165
    # if similarity_type == "asymmetric":
    #     hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
    #     hyperparamethers_range_dictionary["normalize"] = [True]
    #
    # elif similarity_type == "tversky":
    #     hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
    #     hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
    #     hyperparamethers_range_dictionary["normalize"] = [True]
    #
    # if similarity_type in ["cosine", "asymmetric"]:
    #     hyperparamethers_range_dictionary["feature_weighting"] = ["none", "BM25", "TF-IDF"]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM, recommender_list],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"URM_validation": URM_test},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: {"topK": [60, 200, 50, -1], "shrink": [5, 15, -1, -1], # put -1 where useless in order to force you to change when the became useful
                                                               "force_compute_sim": False,
                                                               "old_similarity_matrix": old_similrity_matrix,
                                                               "epochs": 40, "lambda_i": lambda_i, "lambda_j": lambda_j,
                                                               "num_factors": num_factors},
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = this_output_root_path

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize)
    print(best_parameters)


def runParameterSearch_Collaborative(recommender_class, URM_train, ICM=None, metric_to_optimize="MAP",
                                     evaluator_validation=None, evaluator_test=None,
                                     evaluator_validation_earlystopping=None,
                                     output_root_path="result_experiments/", parallelizeKNN=True, n_cases=30,
                                     URM_validation=None):
    from ParameterTuning.AbstractClassSearch import DictionaryKeys

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    try:

        output_root_path_rec_name = output_root_path + recommender_class.RECOMMENDER_NAME

        parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation,
                                         evaluator_test=evaluator_test)

        if recommender_class in [TopPop, Random]:
            recommender = recommender_class(URM_train)

            recommender.fit()

            output_file = open(output_root_path_rec_name + "_BayesianSearch.txt", "a")
            result_dict, result_baseline = evaluator_validation.evaluateRecommender(recommender)
            output_file.write(
                "ParameterSearch: Best result evaluated on URM_validation. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_root_path_rec_name + "_best_result_validation", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            result_dict, result_baseline = evaluator_test.evaluateRecommender(recommender)
            output_file.write("ParameterSearch: Best result evaluated on URM_test. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_root_path_rec_name + "_best_result_test", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            output_file.close()

            return

        ##########################################################################################################

        if recommender_class is UserKNNCFRecommender:

            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=n_cases,
                                                                      output_root_path=output_root_path_rec_name,
                                                                      metric_to_optimize=metric_to_optimize)

            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(4), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

            return

            ##########################################################################################################

        if recommender_class is ItemKNNCFRecommender:

            similarity_type_list = ['cosine']  # , 'jaccard', "asymmetric", "dice", "tversky"]

            # todo n_cases non funziona
            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=1,  # = n_cases
                                                                      output_root_path=output_root_path_rec_name,
                                                                      metric_to_optimize=metric_to_optimize)

            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(4), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

            return

        ##########################################################################################################

        if recommender_class is HybridRecommender:

            similarity_type_list = ['cosine']  # , 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_HybridRecommender_on_similarity_type(),
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      ICM=ICM,
                                                                      n_cases=1,  # = n_cases
                                                                      output_root_path=output_root_path_rec_name,
                                                                      metric_to_optimize=metric_to_optimize)

            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(4), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

            return



            ##########################################################################################################

            # if recommender_class is MultiThreadSLIM_RMSE:
            #
            #     hyperparamethers_range_dictionary = {}
            #     hyperparamethers_range_dictionary["topK"] = [50, 100]
            #     hyperparamethers_range_dictionary["l1_penalty"] = [1e-2, 1e-3, 1e-4]
            #     hyperparamethers_range_dictionary["l2_penalty"] = [1e-2, 1e-3, 1e-4]
            #
            #
            #     recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
            #                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
            #                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
            #                              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
            #                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
            #
            #


            ##########################################################################################################

        if recommender_class is P3alphaRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is RP3betaRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            hyperparamethers_range_dictionary["beta"] = range(0, 2)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["reg"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = [10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["batch_size"] = [1]
            hyperparamethers_range_dictionary["positive_reg"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["negative_reg"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 40, "validation_every_n": 2000,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is PureSVDRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = list(range(0, 250, 5))

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            hyperparamethers_range_dictionary["lambda_i"] = [0.0, 0.1, 0.01, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["lambda_j"] = [0.0, 0.1, 0.01, 1e-3, 1e-6, 1e-9]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights': False,
                                                                               'symmetric': True,
                                                                               'positive_threshold': 1,
                                                                               "URM_validation": URM_validation},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
                                                                       "stop_on_validation": False,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 10,
                                                                       "validation_metric": metric_to_optimize,
                                                                       "epochs": 10},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["l1_penalty"] = [1.0, 0.0, 1e-2, 1e-4, 1e-6]
            hyperparamethers_range_dictionary["l2_penalty"] = [100.0, 1.0, 0.0, 1e-2, 1e-4, 1e-6]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 n_cases=n_cases,
                                                 output_root_path=output_root_path_rec_name,
                                                 metric=metric_to_optimize)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_root_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()


def delete_previous_intermediate_computations():
    '''
    this function is used to remove intermediate old computations that are a problem in new runs
    '''
    files = glob.glob(os.path.join("IntermediateComputations", "*.pkl"))
    for f in files:
        print("Removing file: {}...".format(f))
        os.remove(f)


import os, multiprocessing
from functools import partial


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    # this line removes old matrices saved, comment it if testing only the weights of hybrid
    delete_previous_intermediate_computations()
    dataReader = RS_Data_Loader(top10k=True)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    print("dataset loaded")

    collaborative_algorithm_list = [
        # Random,
        # TopPop,
        # P3alphaRecommender,
        RP3betaRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender#,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender,
        # HybridRecommender
    ]

    from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator_validation_earlystopping = SequentialEvaluator(URM_validation, URM_train, cutoff_list=[10])
    evaluator_test = SequentialEvaluator(URM_test, URM_train, cutoff_list=[10])

    evaluator_validation = EvaluatorWrapper(evaluator_validation_earlystopping)
    evaluator_test = EvaluatorWrapper(evaluator_test)

    multiprocessing_choice = False
    if multiprocessing_choice:

        runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                           URM_train=URM_train,
                                                           ICM=ICM,
                                                           metric_to_optimize="MAP",
                                                           evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                           evaluator_validation=evaluator_validation,
                                                           # evaluator_test=evaluator_test, # I'm not
                                                           # passing it because validation and test for
                                                           # us is the same
                                                           output_root_path=output_root_path,
                                                           n_cases=30)

        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()))
        resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
    else:

        for recommender_class in collaborative_algorithm_list:

            try:

                if recommender_class is HybridRecommender:
                    print("Recommender is an HybridRecommender")
                    recommender_list = [
                        # Random,
                        # TopPop,
                        ItemKNNCBFRecommender,
                        ItemKNNCFRecommender,
                        # P3alphaRecommender,
                        # RP3betaRecommender,
                        # UserKNNCFRecommender,
                        # MatrixFactorization_BPR_Cython,
                        # MatrixFactorization_FunkSVD_Cython,
                        SLIM_BPR_Cython,
                        # SLIMElasticNetRecommender
                        PureSVDRecommender
                    ]

                    if SLIM_BPR_Cython in recommender_list:
                        recommender_IB = ItemKNNCFRecommender(URM_train)
                        recommender_IB.fit(200, 15)
                        transfer_matrix = recommender_IB.W_sparse
                    else:
                        transfer_matrix = None

                    # old similarity matrix is the starting matrix for the SLIM recommender
                    runParameterSearch_Hybrid_partial(recommender_class, URM_train, ICM, recommender_list,
                                                      evaluator_validation=evaluator_validation,
                                                      evaluator_test=evaluator_test, URM_test=URM_test,
                                                      old_similrity_matrix=transfer_matrix)
                else:

                    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                                       URM_train=URM_train,
                                                                       ICM=ICM,
                                                                       metric_to_optimize="MAP",
                                                                       evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                                       evaluator_validation=evaluator_validation,
                                                                       # evaluator_test=evaluator_test, # I'm not
                                                                       # passing it because validation and test for
                                                                       # us is the same
                                                                       output_root_path=output_root_path,
                                                                       n_cases=30,
                                                                       URM_validation=URM_validation)
                    runParameterSearch_Collaborative_partial(recommender_class)

            except Exception as e:

                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()


if __name__ == '__main__':
    read_data_split_and_search()
