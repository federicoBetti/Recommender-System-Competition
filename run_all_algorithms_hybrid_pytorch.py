import sys

from Dataset.RS_Data_Loader import RS_Data_Loader
from KNN.HybridPytorch import HybridPytorch_SLIM
from KNN.HybridSimilaritiesRecommender import HybridSimilaritiesRecommender
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

if __name__ == '__main__':
    evaluate_algorithm = True
    delete_old_computations = False
    slim_after_hybrid = False

    # delete_previous_intermediate_computations()
    # if not evaluate_algorithm:
    #     delete_previous_intermediate_computations()
    # else:
    #     print("ATTENTION: old intermediate computations kept, pay attention if running with all_train")

    filename = "hybrid_one_interval.csv"

    dataReader = RS_Data_Loader(all_train=not evaluate_algorithm, distr_split=True)

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

    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, URM_train, exclude_seen=True)

    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logFile = open(output_root_path + "result_all_algorithms.txt", "a")

    try:
        recommender_class = HybridPytorch_SLIM
        print("Algorithm: {}".format(recommender_class))

        d_weights = []
        onPop = True
        # On pop it used to choose if have dynamic weights for
        recommender = recommender_class(URM_train, ICM, recommender_list, dynamic=False,
                                        d_weights=d_weights, UCM_train=UCM_tfidf,
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

        recommender.fit(**{
            "topK": [10, 220, 150, 160, 61, 276],
            "shrink": [180, 1, 10, 2, -1, -1],
            "pop": [130, 346],
            "weights": [0.5553160795197877, 0.7757009882288872, 0.5982445930826743, 0.5982445930826743,
                        0.9945089580422436, 0.19198495537212745],
            "final_weights": [1, 1],
            "force_compute_sim": True, #not evaluate_algorithm,
            "feature_weighting_index": 0,
            "old_similarity_matrix": old_similrity_matrix,
            "epochs": 50,
            'lambda_i': 0.06928490242552432, 'lambda_j': 0.9408725374123923,
            'alphaP3': [0.5203791059230995],
            'alphaRP3': [0.8182264529058118],
            'betaRP': [0.3775651302605211],
            'l1_ratio': l1_ratio,
            'tfidf': [True, False],
            "weights_to_dweights": -1})

        num_epochs = 2
        for i in range(num_epochs):
            print("Starting Evaluations epoch {} of {}...".format(i, num_epochs))
            results_run, results_run_string, target_recommendations = evaluator.evaluateRecommender(recommender,
                                                                                                    plot_stats=True,
                                                                                                    onPop=onPop)

            print("Algorithm: {}, results: \n{}".format([rec.__class__ for rec in recommender.recommender_list],
                                                        results_run_string))
            recommender.training = False
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

        # if not evaluate_algorithm:
        #     delete_previous_intermediate_computations()
