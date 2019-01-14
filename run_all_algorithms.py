import pickle
import sys

from scipy import sparse

from Dataset.RS_Data_Loader import RS_Data_Loader
from KNN.HybridRecommenderXGBoost import HybridRecommenderXGBoost
from KNN.ItemKNNCFPageRankRecommender import ItemKNNCFPageRankRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

import numpy as np

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.HybridRecommenderTopNapproach import HybridRecommenderTopNapproach
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.HybridRecommender import HybridRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBRecommender
import Support_functions.get_evaluate_data as ged
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import traceback, os

import Support_functions.manage_data as md
from run_parameter_search import delete_previous_intermediate_computations

if __name__ == '__main__':
    evaluate_algorithm = True
    delete_old_computations = False
    slim_after_hybrid = False

    # delete_previous_intermediate_computations()
    # if not evaluate_algorithm:
    #     delete_previous_intermediate_computations()
    # else:
    #     print("ATTENTION: old intermediate computations kept, pay attention if running with all_train")

    filename = "hybrid_different_rec_for_diff_intervals_150.csv"

    dataReader = RS_Data_Loader(all_train=not evaluate_algorithm)

    URM_train = dataReader.get_URM_train()
    URM_PageRank_train = dataReader.get_page_rank_URM()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    UCM_tfidf = dataReader.get_tfidf_artists()
    # _ = dataReader.get_tfidf_album()

    # URM_train = dataReader.get_page_rank_URM()
    #
    # ITEMB
    # CB, ITEM
    # CF, USER
    # CF, P3ALPHA, RP3BETA, PURE
    # SVD
    recommender_list1 = [
        # Random,
        # TopPop,
        ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        # ItemKNNCFPageRankRecommender,
        # P3alphaRecommender,
        RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        SLIM_BPR_Cython,
        # PureSVDRecommender,
        SLIMElasticNetRecommender
    ]

    # ITEM CB, ITEM CF, USER CF, RP3BETA, PURE SVD
    recommender_list2 = [
        # Random,
        # TopPop,
        ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        # ItemKNNCFPageRankRecommender,
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

    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, URM_train, exclude_seen=True)

    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logFile = open(output_root_path + "result_all_algorithms.txt", "a")

    try:
        recommender_class = HybridRecommender
        print("Algorithm: {}".format(recommender_class))

        '''
        Our optimal run
        '''
        recommender_list = recommender_list1  # + recommender_list2  # + recommender_list3

        onPop = True

        # On pop it used to choose if have dynamic weights for
        recommender = recommender_class(URM_train, ICM, recommender_list, URM_PageRank_train=URM_PageRank_train,
                                        dynamic=False, UCM_train=UCM_tfidf,
                                        URM_validation=URM_validation, onPop=onPop)

        lambda_i = 0.1
        lambda_j = 0.05
        old_similrity_matrix = None
        num_factors = 395
        l1_ratio = 1e-06

        # Variabili secondo intervallo
        alphaRP3_2 = 0.9223827655310622
        betaRP3_2 = 0.2213306613226453
        num_factors_2 = 391

        recommender.fit(**
                        {
                            "topK": [130, 170, 180, 391, 761, 480],
                            "shrink": [2, 2, 2, -1, -1, -1],
                            "pop": [280],
                            "weights": [0.36503223650050454, 0.83112926076437,  0.7930753751595545,  0.6847526619355242, 0.2418366304926498, 0.9966317374868113],
                            "force_compute_sim": False,
                            "feature_weighting_index": 1,
                            "epochs": 50,
                            'lambda_i': [0.0], 'lambda_j': [1.0153577332223556e-08], 'SLIM_lr': [0.1],
                            'alphaP3': [0.7649722376036994],
                            'alphaRP3': [0.8582865731462926],
                            'betaRP': [0.2814208416833668],
                            'alpha': 0.0014681984611695231,
                            'l1_ratio': 3.020408163265306e-06,
                            "weights_to_dweights": -1})

        print("TEST")
        print("Starting Evaluations...")
        # to indicate if plotting for lenght or for pop

        results_run, results_run_string, target_recommendations = evaluator.evaluateRecommender(recommender,
                                                                                                plot_stats=True,
                                                                                                onPop=False)

        print("Algorithm: {}, results: \n{}".format([rec.__class__ for rec in recommender.recommender_list],
                                                    results_run_string))
        logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
        logFile.flush()


        if not evaluate_algorithm:
            target_playlist = dataReader.get_target_playlist()
            md.assign_recomendations_to_correct_playlist(target_playlist, target_recommendations)
            md.make_CSV_file(target_playlist, filename)
            print('File {} created!'.format(filename))


    except Exception as e:
        traceback.print_exc()
        logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
        logFile.flush()

    if not evaluate_algorithm:
        delete_previous_intermediate_computations()
