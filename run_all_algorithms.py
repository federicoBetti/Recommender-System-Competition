import sys

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
import Support_functions.get_evaluate_data as ged
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender

from data.Movielens_10M.Movielens10MReader import Movielens10MReader

import traceback, os

import Support_functions.manage_data as md
from run_parameter_search import delete_previous_intermediate_computations

if __name__ == '__main__':
    evaluate_algorithm = False
    slim_after_hybrid = False

    delete_previous_intermediate_computations()

    filename = "hybrid_4algorithms_P3_thresholds: 130, 346.csv"

    dataReader = RS_Data_Loader(slim_after_hybrid, top10k=True, all_train=not evaluate_algorithm)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()

    recommender_list = [
        # Random,
        # TopPop,
        # RP3betaRecommender,
        ItemKNNCBFRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # SLIM_BPR_Cython,
        # # SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    weights = [
        1,
        2,
        3,
        4
    ]

    topK = [
        60,
        100,
        150,
        50
    ]

    shrinks = [
        5,
        50,
        10,
        0
    ]

    # For hybrid with weighted estimated rating
    d_weights = [[0.5036233341368713, 0.33188050208014197, 0.011954128257236074, 0.9989615570048721],
                 [0.11103826438392561, 0.028116683377075735, 0.024329672375916767, 0.6743664705280373],
                 [0.8451197847820727, 0.9253649964257065, 0.3313925225051185, 0.7574371680908533]]


    pop = [130, 346]
    # BEST RESULT : d_weights = [[0.5, 0.5, 0], [0.4, 0.4, 0.2], [0, 0.8, 0.2], [0, 0.5, 0.5]]

    # Dynamics for Hybrid with Top_N. usefull for testing where each recommender works better
    #d_weights = [[2, 4, 0], [1, 4, 5], [0, 2, 8]]

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

        recommender = recommender_class(URM_train, ICM, recommender_list, dynamic=True, d_weights=d_weights,
                                        URM_validation=URM_validation)
        recommender.fit(**{"topK": topK, "shrink": shrinks, "force_compute_sim": False,
                                                               "old_similarity_matrix": None,
                                                               "epochs": 150, "lambda_i": 0.01, "lambda_j": 0.01,
                                                               "num_factors": 165, "weights": weights, "pop": pop})

        print("Starting Evaluations...")
        results_run, results_run_string, target_recommendations = evaluator.evaluateRecommender(recommender,
                                                                                                plot_stats=True)

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

    delete_previous_intermediate_computations()
