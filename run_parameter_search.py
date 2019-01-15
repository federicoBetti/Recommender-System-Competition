#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
import glob

from Base.Evaluation.Evaluator import SequentialEvaluator, ParallelEvaluator
from Base.NonPersonalizedRecommender import TopPop, Random
from Dataset.RS_Data_Loader import RS_Data_Loader
from KNN.HybridRecommender import HybridRecommender, HybridRecommender_Test_Not_Weights
from KNN.HybridSimilaritiesRecommender import HybridSimilaritiesRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFPageRankRecommender import ItemKNNCFPageRankRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
# from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
from MatrixFactorization.MatrixFactorization_RMSE import IALS_numpy
from ParameterTuning.RandomSearch import RandomSearch
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_ElasticNet
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from Support_functions import get_evaluate_data as ged

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from ParameterTuning.BayesianSearch import BayesianSearch

import traceback, pickle
from Utils.PoolWithSubprocess import PoolWithSubprocess

from ParameterTuning.AbstractClassSearch import DictionaryKeys
import numpy as np


def run_KNNCFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, n_cases, output_root_path,
                                            metric_to_optimize, UCM_train=None):
    # pay attention that it doesn't finish (it should after n_cases, but now it doens't work)
    # it saves best results in the txt file
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = list(range(40, 250, 5))
    hyperparamethers_range_dictionary["shrink"] = list(range(0, 50, 1))
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    if UCM_train is None:  # Collaborative Filtering Algorithms
        first_dict = [URM_train]
        hyperparamethers_range_dictionary["tfidf"] = [True, False]
    else:  # Content Base Filtering Algorithms
        first_dict = [UCM_train, URM_train]
        hyperparamethers_range_dictionary["feature_weighting_index"] = [0, 1, 2]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: first_dict,
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),  # questi sono quelli fissi
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}  # questi sono quelli che stai testando

    output_root_path_similarity = output_root_path + "_" + similarity_type

    # questo runna fit del recommender
    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             init_points=30,
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
                                      old_similrity_matrix=None, UCM_train=None, URM_pagerank=None):
    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    ##########################################################################################################

    recommender_list = [
        # Random,
        # TopPop,
        ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        # ItemKNNCFPageRankRecommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        IALS_numpy,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    this_output_root_path = output_root_path + "/Hybrid_top:" + "{}".format(
        "_".join([x.RECOMMENDER_NAME for x in recommender_list]))

    # since test and validation are the same for now, here I don't pass the evaluator test (otherwise it also crash)
    parameterSearch = BayesianSearch(recommender_class, evaluator_validation)

    hyperparamethers_range_dictionary = {}
    weights = [0.5017146067001184, 1.297666068549087, 0.7554431209026025, 1.661705677455121,
               0.20789770603801933, 2.1198458359150587]
    hyperparamethers_range_dictionary["weights1"] = list(np.linspace(weights[0] - 0.15, weights[0] + 0.15, 300))
    hyperparamethers_range_dictionary["weights2"] = list(np.linspace(weights[1] - 0.15, weights[1] + 0.15, 300))
    hyperparamethers_range_dictionary["weights3"] = list(np.linspace(weights[2] - 0.15, weights[2] + 0.15, 300))
    hyperparamethers_range_dictionary["weights4"] = list(np.linspace(weights[3] - 0.15, weights[3] + 0.15, 300))
    hyperparamethers_range_dictionary["weights5"] = list(np.linspace(weights[4] - 0.15, weights[4] + 0.15, 300))
    hyperparamethers_range_dictionary["weights6"] = list(np.linspace(weights[5] - 0.15, weights[5] + 0.15, 300))

    # hyperparamethers_range_dictionary["weights1"] = range(0, 1)
    # hyperparamethers_range_dictionary["weights2"] = range(0, 2)
    # hyperparamethers_range_dictionary["weights3"] = range(0, 2)
    # hyperparamethers_range_dictionary["weights4"] = range(0, 2)
    # hyperparamethers_range_dictionary["weights5"] = range(0, 2)
    # hyperparamethers_range_dictionary["weights6"] = range(0, 2)
    # hyperparamethers_range_dictionary["weights7"] = range(0, 1)
    # hyperparamethers_range_dictionary["weights8"] = range(0, 1)

    # hyperparamethers_range_dictionary["filter_top_pop_len"] = list(range(0, 20))

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM, recommender_list],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"URM_validation": URM_test, "dynamic": False,
                                                                       "UCM_train": UCM_train,
                                                                       "URM_PageRank_train": URM_pagerank
                                                                       },
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: {
                                 "topK": [10, 33, 160, -1, 761, 490],
                                 "shrink": [8, 26, 2, -1, -1, -1],
                                 "pop": [280],
                                 "weights": [1, 1, 1, 1, 1, 1],
                                 "final_weights": [1, 1],
                                 "force_compute_sim": False,  # not evaluate_algorithm,
                                 "feature_weighting_index": 0,
                                 "epochs": 150,
                                 'lambda_i': [0.0], 'lambda_j': [1.0153577332223556e-08], 'SLIM_lr': [0.1],
                                 'alphaP3': [0.6065173289897832],
                                 'alphaRP3': [0.8582865731462926],
                                 'betaRP': [0.2814208416833668],
                                 'l1_ratio': 3.020408163265306e-06,
                                 'alpha': 0.0014681984611695231,
                                 'tfidf': [True],
                                 "weights_to_dweights": -1,
                                 "IALS_num_factors": 290,
                                 "IALS_reg": 0.001,
                                 "IALS_iters": 6,
                                 "IALS_scaling": 'log',
                                 "IALS_alpha": 40,
                                 # "filter_top_pop_len": 0
                             },
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = this_output_root_path

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=90,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize,
                                             init_points=60
                                             )
    print(best_parameters)


def runParameterSearch_Hybrid_partial_single(recommender_class, URM_train, ICM, recommender_list, n_cases=35,
                                             evaluator_validation=None, evaluator_test=None, metric_to_optimize="MAP",
                                             output_root_path="result_experiments/", parallelizeKNN=False,
                                             URM_test=None, URM_page_rank=None,
                                             old_similrity_matrix=None, UCM_train=None):
    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)
    recommender_class = HybridRecommender_Test_Not_Weights
    ##########################################################################################################

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
        SLIM_BPR_Cython,
        # ItemKNNCFRecommenderFAKESLIM,
        SLIMElasticNetRecommender,
        PureSVDRecommender
    ]

    this_output_root_path = output_root_path + "single_rec_test:" + "{}".format(
        "_".join([x.RECOMMENDER_NAME for x in recommender_list]))

    # since test and validation are the same for now, here I don't pass the evaluator test (otherwise it also crash)
    parameterSearch = BayesianSearch(recommender_class, evaluator_validation)

    hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["top1"] = list(range(0, 150, 2))  # [130]
    # hyperparamethers_range_dictionary["alphaRP3"] = range(0, 2)  # [130]
    # hyperparamethers_range_dictionary["betaRP"] = range(0, 2)  # [130]
    # hyperparamethers_range_dictionary["top2"] = [170]  # list(range(0, 300, 5))  # [170]
    # hyperparamethers_range_dictionary["top3"] = list(range(20, 250, 2))  # [160]
    # hyperparamethers_range_dictionary["top4"] = [160]  # list(range(0, 300, 5))  # [160]
    # hyperparamethers_range_dictionary["top5"] = [160]  # list(range(0, 300, 5))  # [160]
    # hyperparamethers_range_dictionary["top6"] = [761]
    # hyperparamethers_range_dictionary["top7"] = [490]
    hyperparamethers_range_dictionary["num_factors"] = list(range(250, 600, 2))
    hyperparamethers_range_dictionary["n_iter"] = list(range(1, 5, 1))

    # hyperparamethers_range_dictionary["shrink1"] = list(range(0, 50, 1))  # [2]
    # hyperparamethers_range_dictionary["feature_weighting_index"] = [0, 1, 2]
    # hyperparamethers_range_dictionary["shrink3"] = list(range(0, 10, 1))  # [2]
    # hyperparamethers_range_dictionary["shrink3"] = [2]  # list(range(0, 50, 1))  # [1]
    # hyperparamethers_range_dictionary["shrink4"] = [-1]
    # hyperparamethers_range_dictionary["shrink5"] = [-1]
    # hyperparamethers_range_dictionary["shrink6"] = [-1]
    # hyperparamethers_range_dictionary["shrink7"] = [-1]
    # hyperparamethers_range_dictionary["shrink6"] = [-1]
    # hyperparamethers_range_dictionary["tfidf"] = [True] # [2]


    # METTERE COME PARAMETRI QUELLI CHE SI VOGLIONO TESTARE SOPRA E COMMENTARLI SOTTO
    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM, recommender_list],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"URM_validation": URM_test, "dynamic": False,
                                                                       "UCM_train": UCM_train},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: {
                                 "top1": 10,
                                 "top2": 33,
                                 "top3": 160,
                                 # "top4": 5,
                                 # "top5": 82,
                                 "top5": 761,
                                 "top6": 490,
                                 "top7": -1,
                                 "shrink1": 8,
                                 "shrink2": 26,
                                 "shrink3": 2,
                                 # "shrink4": -1,
                                 "shrink5": -1,
                                 "shrink6": -1,
                                 "shrink7": -1,
                                 "pop": [280],
                                 "weights": [0.838635347569101, 1.9502084787559169, 1.4864947454516366,
                                             0.43913055657214306, 1.287881329182088, 1.5249378222590844],
                                 "final_weights": [1, 1],
                                 "force_compute_sim": False,  # not evaluate_algorithm,
                                 "feature_weighting_index": 1,
                                 "epochs": 150,
                                 'lambda_i': [0.0], 'lambda_j': [1.0153577332223556e-08], 'SLIM_lr': [0.1],
                                 'alphaP3': [0.4121720883248633],
                                 'alphaRP3': [1.4195316958336883],
                                 'betaRP': [0.27677052728067486],
                                 'l1_ratio': 3.020408163265306e-06,
                                 'alpha': 0.0014681984611695231,
                                 'tfidf': [True],
                                 "weights_to_dweights": -1,
                                 "filter_top_pop_len": 0},
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = this_output_root_path

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=90,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize,
                                             init_points=40
                                             )
    print(best_parameters)


def runParameterSearch_Collaborative(recommender_class, URM_train, ICM=None, metric_to_optimize="MAP",
                                     evaluator_validation=None, evaluator_test=None,
                                     evaluator_validation_earlystopping=None,
                                     output_root_path="result_experiments/", parallelizeKNN=False, n_cases=30,
                                     URM_validation=None, UCM_train=None, URM_page_rank=None):
    from ParameterTuning.AbstractClassSearch import DictionaryKeys

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    try:

        output_root_path_rec_name = output_root_path + recommender_class.RECOMMENDER_NAME

        parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation)

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

            similarity_type_list = ['cosine']  # , 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=40,
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

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=40,  # = n_cases
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
        ##########################################################################################################


        if recommender_class is ItemKNNCFPageRankRecommender:
            similarity_type_list = ['cosine']  # , 'jaccard', "asymmetric", "dice", "tversky"]

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = list(range(30, 400, 10))
            hyperparamethers_range_dictionary["shrink"] = list(range(0, 100, 2))

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, URM_page_rank],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),  # questi sono quelli fissi
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

            ##########################################################################################################
            ##########################################################################################################

        if recommender_class is ItemKNNCBFRecommender:

            similarity_type_list = ['cosine']  # , 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=30,
                                                                      output_root_path=output_root_path_rec_name,
                                                                      metric_to_optimize=metric_to_optimize,
                                                                      UCM_train=ICM)

            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(4), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

            return

            ##########################################################################################################

        if recommender_class is UserKNNCBRecommender:

            similarity_type_list = ['cosine']  # , 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=50,
                                                                      output_root_path=output_root_path_rec_name,
                                                                      metric_to_optimize=metric_to_optimize,
                                                                      UCM_train=UCM_train)

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
            hyperparamethers_range_dictionary["topK"] = list(range(1, 800, 5))
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
            n_cases = 50

        ##########################################################################################################

        if recommender_class is RP3betaRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = list(range(1, 800, 5))
            hyperparamethers_range_dictionary["alpha"] = list(np.linspace(0.001, 2.0, 500))  # range(0, 2)
            hyperparamethers_range_dictionary["beta"] = list(np.linspace(0.001, 2.0, 500))  # range(0, 2) np.linespace()
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is IALS_numpy:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = list(range(10, 300, 20))
            hyperparamethers_range_dictionary["reg"] = list(np.linspace(0.001, 0.3, 200))  # range(0, 2)
            # hyperparamethers_range_dictionary["iters"] = list(range(5, 50, 2))  # range(0, 2) np.linespace()
            hyperparamethers_range_dictionary["scaling"] = ['linear', 'log']
            hyperparamethers_range_dictionary["alpha"] = list(range(10, 100, 5))

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
            n_cases = 60

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
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad"]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = [10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["batch_size"] = [1000]
            hyperparamethers_range_dictionary["positive_reg"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["negative_reg"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [0.1, 1e-2, 1e-3, 1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 1000, "validation_every_n": 500,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_AsySVD_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad"]  # , "adam"]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = list(range(10, 200, 10))
            # hyperparamethers_range_dictionary["batch_size"] = [1]
            hyperparamethers_range_dictionary["positive_reg"] = range(0, 1)
            hyperparamethers_range_dictionary["negative_reg"] = range(0, 1)
            hyperparamethers_range_dictionary["learning_rate"] = range(0, 1)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 40, "validation_every_n": 2000,
                                                                       "stop_on_validation": False,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################
        '''
        if recommender_class is MF_MSE_PyTorch:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = list(range(10, 200, 10))
            hyperparamethers_range_dictionary["learning_rate"] = [
                0.001]  # [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 40, "validation_every_n": 20,
                                                                       "stop_on_validation": False,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
        '''
        ##########################################################################################################

        if recommender_class is PureSVDRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = list(range(0, 400, 5))
            hyperparamethers_range_dictionary["n_iter"] = list(range(0, 50))

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = list(range(1, 800, 40))
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            # hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            hyperparamethers_range_dictionary["lambda_i"] = range(0, 1)
            hyperparamethers_range_dictionary["lambda_j"] = range(0, 1)
            hyperparamethers_range_dictionary["learning_rate"] = [0.1, 1e-2, 1e-3, 1e-4, 1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights': False,
                                                                               'symmetric': True,
                                                                               'positive_threshold': 1,
                                                                               "URM_validation": URM_validation},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 500,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 2,
                                                                       "validation_metric": metric_to_optimize,
                                                                       "epochs": 30},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:
            parameterSearch = BayesianSearch(recommender_class, evaluator_validation)
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = list(range(10, 500, 30))
            hyperparamethers_range_dictionary["l1_ratio"] = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
            n_cases = 20

        if recommender_class is MultiThreadSLIM_ElasticNet:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = list(range(10, 500, 30))
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            hyperparamethers_range_dictionary["l1_ratio"] = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

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
                                                 metric=metric_to_optimize,
                                                 init_points=40
                                                 )



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
    # delete_previous_intermediate_computations()
    dataReader = RS_Data_Loader(distr_split=True)  # random train and test

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    URM_pagerank = dataReader.get_page_rank_URM()
    ICM = dataReader.get_ICM()
    UCM_train = dataReader.get_tfidf_artists()
    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    print("dataset loaded")

    collaborative_algorithm_list = [
        # Random,
        # TopPop,
        # ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # ItemKNNCFPageRankRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MF_MSE_PyTorch,
        # MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender,
        # MultiThreadSLIM_ElasticNet,
        # IALS_numpy,
        HybridRecommender
    ]

    # if UserKNNCBRecommender in collaborative_algorithm_list:
    #     ICM = dataReader.get_tfidf_artists()
    #     ICM = dataReader.get_tfidf_album()
    # elif ItemKNNCBFRecommender in collaborative_algorithm_list:
    #     ICM = dataReader.get_ICM()

    from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator_validation_earlystopping = SequentialEvaluator(URM_validation, URM_train, cutoff_list=[10])
    evaluator_test = SequentialEvaluator(URM_test, URM_train, cutoff_list=[10])
    # evaluator_validation_earlystopping = ParallelEvaluator(URM_test, cutoff_list=[10])
    # evaluator_test = ParallelEvaluator(URM_test, cutoff_list=[10])

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
                        UserKNNCBRecommender,
                        ItemKNNCFPageRankRecommender,
                        ItemKNNCFRecommender,
                        UserKNNCFRecommender,
                        # P3alphaRecommender,
                        RP3betaRecommender,
                        # MatrixFactorization_BPR_Cython,
                        # MatrixFactorization_FunkSVD_Cython,
                        SLIM_BPR_Cython,
                        SLIMElasticNetRecommender
                        # PureSVDRecommender
                    ]

                    if SLIM_BPR_Cython in recommender_list:
                        recommender_IB = ItemKNNCFRecommender(URM_train)
                        recommender_IB.fit(200, 15)
                        transfer_matrix = recommender_IB.W_sparse
                    else:
                        transfer_matrix = None

                    # runna single algorithm
                    single = False
                    if single is False:
                        # old similarity matrix is the starting matrix for the SLIM recommender
                        runParameterSearch_Hybrid_partial(recommender_class, URM_train, ICM, recommender_list,
                                                          URM_pagerank=URM_pagerank,
                                                          evaluator_validation=evaluator_validation,
                                                          evaluator_test=evaluator_test, URM_test=URM_test,
                                                          old_similrity_matrix=transfer_matrix, UCM_train=UCM_train)
                    else:
                        runParameterSearch_Hybrid_partial_single(recommender_class, URM_train, ICM, recommender_list,
                                                                 evaluator_validation=evaluator_validation,
                                                                 evaluator_test=evaluator_test, URM_test=URM_test,
                                                                 old_similrity_matrix=transfer_matrix,
                                                                 UCM_train=UCM_train, URM_page_rank=URM_pagerank)
                else:

                    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                                       URM_train=URM_train,
                                                                       ICM=ICM,
                                                                       UCM_train=UCM_train,
                                                                       metric_to_optimize="MAP",
                                                                       evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                                       evaluator_validation=evaluator_validation,
                                                                       output_root_path=output_root_path,
                                                                       n_cases=60,
                                                                       URM_validation=URM_validation,
                                                                       URM_page_rank=URM_pagerank)
                    runParameterSearch_Collaborative_partial(recommender_class)

            except Exception as e:

                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()

            print("ciao")


if __name__ == '__main__':
    read_data_split_and_search()
