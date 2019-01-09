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
        # ItemKNNCBFRecommender,
        ItemKNNCFPageRankRecommender,
        # UserKNNCBRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    # ITEM CB, ITEM CF, USER CF, RP3BETA, PURE SVD
    recommender_list2 = [
        # Random,
        # TopPop,
        # ItemKNNCBFRecommender,
        # ItemKNNCFPageRankRecommender,
        # UserKNNCBRecommender,
        ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    # UserCBF, ItemCF, UserCF, P3alpha, RP3b, SLIM, PurSVD
    recommender_list3 = [
        # Random,
        # TopPop,
        # ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # # MatrixFactorization_BPR_Cython,
        # # MatrixFactorization_FunkSVD_Cython,
        # SLIM_BPR_Cython,
        # # SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    # For hybrid with weighted estimated rating
    d_weights = [
        [0.6708034395599534, 0.4180455311930482, 4180455311930482, 0.013121631586130333, 0.9606783176615321,
         0.9192576193987754] + [0] * len(recommender_list2) + [0] * len(recommender_list3),
        [0] * len(recommender_list1) + [0.03206429006541767, 0.022068399812202766, 0.5048937312439359,
                                        0.5777889378285606, 0.002469536740713263, 2959761085665614]
        + [0] * len(recommender_list3),
        [0] * len(recommender_list1) + [0] * len(recommender_list2) + [0.2959761085665614, 0.08296490886624563,
                                                                       0.72672714096492, 0.04856215067017522,
                                                                       0.7144382800343254, 0.20367609381116258,
                                                                       0.1080480529784491]
    ]
    #
    # d_best = [[0.4, 0.03863232277574469, 0.008527738266632112, 0.2560912624445676, 0.7851755932819731,
    #            0.4112843940329439],
    #           [0.2, 0.012499871230102988, 0.020242981888115352, 0.9969708006657074, 0.9999132876156388,
    #            0.6888103295594851],
    #           [0.2, 0.10389111810225915, 0.14839466129917822, 0.866992903043857, 0.07010619211847613,
    #            0.5873532658846817]]

    # BEST RESULT : d_weights = [[0.5, 0.5, 0], [0.4, 0.4, 0.2], [0, 0.8, 0.2], [0, 0.5, 0.5]]

    # Dynamics for Hybrid with Top_N. usefull for testing where each recommender works better
    # d_weights = [[2, 4, 0], [1, 4, 5], [0, 2, 8]]

    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, URM_train, exclude_seen=True)

    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logFile = open(output_root_path + "result_all_algorithms.txt", "a")

    transfer_learning = False
    if transfer_learning:
        recommender_IB = ItemKNNCFRecommender(URM_train)
        recommender_IB.fit(200, 15)
        transfer_matrix = recommender_IB.W_sparse
    else:
        transfer_matrix = None

    try:
        recommender_class = HybridRecommender
        print("Algorithm: {}".format(recommender_class))

        '''
        Test run with hybrid of hybrids
        '''
        # onPop = True
        # recommender_1 = recommender_class(URM_train, ICM, recommender_list, UCM_train=UCM_tfidf, dynamic=True,
        #                                   d_weights=d_weights,
        #                                   URM_validation=URM_validation, onPop=onPop)
        #
        # onPop = False
        # recommender_2 = recommender_class(URM_train, ICM, recommender_list, UCM_train=UCM_tfidf, dynamic=True,
        #                                   d_weights=d_weights,
        #                                   URM_validation=URM_validation, onPop=onPop)
        #
        # pop = [130, 346]
        # recommender_1.fit(**{"topK": topK,
        #                      "shrink": shrinks,
        #                      "pop": pop,
        #                      # "pop": [130, 346],
        #                      "weights": [1, 1],
        #                      # put -1 where useless in order to force you to change when the became useful
        #                      "force_compute_sim": True,
        #                      'alphaP3': 0.6048420766420062,
        #                      'alphaRP3': 1.5890147620983466,
        #                      'betaRP': 0.28778362462762974})
        #
        # pop = [15, 30]
        # recommender_2.fit(**{"topK": topK,
        #                      "shrink": shrinks,
        #                      "pop": pop,
        #                      # "pop": [130, 346],
        #                      "weights": [1, 1],
        #                      # put -1 where useless in order to force you to change when the became useful
        #                      "force_compute_sim": True,
        #                      'alphaP3': 0.6048420766420062,
        #                      'alphaRP3': 1.5890147620983466,
        #                      'betaRP': 0.28778362462762974})
        #
        # recommender_list_2 = [recommender_1, recommender_2]
        # recommender = recommender_class(URM_train, ICM, recommender_list_2, UCM_train=UCM_tfidf, dynamic=False,
        #                                 d_weights=d_weights, weights=weights,
        #                                 URM_validation=URM_validation, onPop=onPop, moreHybrids=False)
        #

        '''
        Our optimal run
        '''
        recommender_list = recommender_list1 #+ recommender_list2# + recommender_list3
        onPop = True


        # On pop it used to choose if have dynamic weights for
        recommender = recommender_class(URM_train, ICM, recommender_list, URM_PageRank_train=URM_PageRank_train,
                                        dynamic=False,
                                        d_weights=d_weights, UCM_train=UCM_tfidf,
                                        URM_validation=URM_validation, onPop=onPop, )#tracks=dataReader.tracks)

        # dtrain = xgb.DMatrix(URM_train, label=)
        # dtest = xgb.DMatrix(X_test, label=y_test)

        lambda_i = 0.1
        lambda_j = 0.05
        old_similrity_matrix = None
        num_factors = 395
        l1_ratio = 1e-06

        # Variabili secondo intervallo
        alphaRP3_2 = 0.9223827655310622
        betaRP3_2 = 0.2213306613226453
        num_factors_2 = 391

        # UserCBF, ItemCF, UserCF, P3alpha, RP3b, SLIM, PurSVD
        #
        # Item
        # Collaborative: Best
        # config is: Config
        # {'top1': 180, 'shrink1': 2}, MAP

        # User
        # Collaborative: Best
        # config is: Config
        # {'top1': 240, 'shrink1': 19}, MAP

        # PureSVD: Best
        # config is: Config: {'num_factors': 95} - MAP

        # P3Beta: Best
        # config is: Config
        # {'shrink1': 80, 'top1': 151, 'alphaP3': 1.2827139967773968, 'normalize_similarity': False}, MAP

        # RP3Beta: Best
        # config is: Config
        # {'top1': 91, 'shrink1': -1, 'alphaRP3': 0.49774549098196397, 'betaRP': 0.2333486973947896}, MAP

        # SLIM_BPR: Best
        # config is: Config
        # {'top1': 311, 'lambda_i': 0.10467537896611145, 'lambda_j': 0.004454204678491891, 'shrink1': -1}, MAP

        # Item
        # Content: Schifo

        # User
        # Content: {'top1': 250, 'shrink1': 55, 'normalize': False} - MAP

        # ElasticNet: New
        # best
        # config
        # found.Config: {'top1': 50, 'l1_ratio': 1e-06, 'shrink1': -1} - MAP

        recommender.fit(**{
            "topK": [15,],# 595, 400, 105, 15, 20],#+ [21, 220, 300, 160, 70, -1],# + [250, 180, 240, 151, 91, 311, -1],
            "shrink": [210],# 1, 1, 30, -1, -1],# + [75, 1, 1, 150, -1, -1],# + [55, 2, 19, -1, -1, -1, -1],
            "pop": [350],
            "weights": [1] * 1,
            "force_compute_sim": True,
            # "feature_weighting_index": 0,
            "old_similarity_matrix": old_similrity_matrix,
            "epochs": 1, "lambda_i": [0.10467537896611145],
            "lambda_j": [0.004454204678491891],  # SOLO ULTIMO HA SLIM
            "num_factors": [395, 391, 95],
            'alphaP3': [0.7100641282565131, 1.2827139967773968],
            'alphaRP3': [0.457685370741483, 0.9223827655310622, 0.49774549098196397],
            'betaRP': [0.289432865731463, 0.2213306613226453, 0.2333486973947896],
            'l1_ratio': l1_ratio,
            'feature_weighting_index': 1,
            "weights_to_dweights":-1})

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
