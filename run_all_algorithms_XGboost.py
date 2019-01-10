import pickle
import sys

from scipy import sparse
from sklearn.datasets import dump_svmlight_file

from Dataset.RS_Data_Loader import RS_Data_Loader
from KNN.HybridRecommenderXGBoost import HybridRecommenderXGBoost, HybridRecommenderXGBoost_Fede
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
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
from Base.Evaluation.Evaluator import SequentialEvaluator

from data.Movielens_10M.Movielens10MReader import Movielens10MReader

import traceback, os

import Support_functions.manage_data as md
from run_parameter_search import delete_previous_intermediate_computations

if __name__ == '__main__':
    evaluate_algorithm = True
    delete_old_computations = False
    slim_after_hybrid = False
    XGB_model_ready = True

    # delete_previous_intermediate_computations()
    # if not evaluate_algorithm:
    #     delete_previous_intermediate_computations()
    # else:
    #     print("ATTENTION: old intermediate computations kept, pay attention if running with all_train")

    filename = "hybrid_different_rec_for_diff_intervals_150.csv"

    dataReader = RS_Data_Loader(all_train=not evaluate_algorithm)

    URM_train = dataReader.get_URM_train()
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
    recommender_list = [
        # Random,
        # TopPop,
        ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    # For hybrid with weighted estimated rating
    d_weights = [
        0.6708034395599534, 0.4180455311930482, 0.013121631586130333, 0.9606783176615321, 0.9192576193987754
    ]

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
        recommender_class = HybridRecommenderXGBoost_Fede
        print("Algorithm: {}".format(recommender_class))

        onPop = True

        # XGBoost
        model = None

        recommender = recommender_class(URM_train, ICM, recommender_list, tracks=dataReader.tracks,
                                        XGB_model_ready=False,
                                        XGBoost_model=model,
                                        dynamic=False,
                                        d_weights=d_weights, UCM_train=UCM_tfidf,
                                        URM_validation=URM_validation, onPop=onPop)

        # dtrain = xgb.DMatrix(URM_train, label=)
        # dtest = xgb.DMatrix(X_test, label=y_test)

        recommender.fit(**{
            "topK": [15, 595, 105, 15, 20],
            "shrink": [210, 1, 30, -1, -1],
            "pop": [130, 346],
            "weights": d_weights,
            "force_compute_sim": False,
            "feature_weighting_index": 0,
            "old_similarity_matrix": None,
            "epochs": 1, "lambda_i": [0.10467537896611145],
            "lambda_j": [0.004454204678491891],  # SOLO ULTIMO HA SLIM
            "num_factors": [395, 391, 95],
            'alphaP3': [0.7100641282565131, 1.2827139967773968],
            'alphaRP3': [0.457685370741483, 0.9223827655310622, 0.49774549098196397],
            'betaRP': [0.289432865731463, 0.2213306613226453, 0.2333486973947896],
            'l1_ratio': 123,
            "weights_to_dweights": -1})

        print("Starting Evaluations...")
        # to indicate if plotting for lenght or for pop

        results_run, results_run_string, target_recommendations = evaluator.evaluateRecommender(recommender,
                                                                                                plot_stats=False,
                                                                                                onPop=onPop)

        print("Algorithm without XGboost: {}, results: \n{}".format(
            [rec.__class__ for rec in recommender.recommender_list],
            results_run_string))

        '''
        train the model and rerun the evaluation
        '''
        X_train = recommender.trainXGBoost
        Y_train = recommender.label_XGboost
        print("XGBoost training...")
        print("X_train type: {}, shape: {}".format(type(X_train), X_train.shape))
        print("y_train shape: {}".format(Y_train.shape))

        # dump_svmlight_file(X_train, Y_train, 'dtrain.svm', zero_based=True)
        # dtrain = xgb.DMatrix('dtrain.svm')
        dtrain = xgb.DMatrix(X_train, label=Y_train)

        param = {
            'max_depth': 5,  # the maximum depth of each tree
            'eta': 1,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'binary:logistic',  # error evaluation for multiclass training
            'num_class': 1   # the number of classes that exist in this datset
        }
        num_round = 60  # the number of training iterations

        model = xgb.train(param, dtrain, num_round, verbose_eval=2)

        recommender.xgb_model_ready = True
        recommender.xgbModel = model

        results_run, results_run_string, target_recommendations = evaluator.evaluateRecommender(recommender,
                                                                                                plot_stats=False,
                                                                                                onPop=onPop)

        print("Algorithm with XGboost: {}, results: \n{}".format(
            [rec.__class__ for rec in recommender.recommender_list],
            results_run_string))

        logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
        logFile.flush()

        if not XGB_model_ready:
            with open(os.path.join("Dataset", "XGBoostTraining.pkl"), 'wb') as handle:
                pickle.dump(recommender.trainXGBoost, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join("Dataset", "XGBoostLabels.pkl"), 'wb') as handle:
                pickle.dump(recommender.user_id_XGBoost, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
