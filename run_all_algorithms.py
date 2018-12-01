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
    slim_after_hybrid = False

    #delete_previous_intermediate_computations()

    filename = "hybrid_UserContentMatrix"

    dataReader = RS_Data_Loader(top10k=True, all_train=not evaluate_algorithm)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    UCM_tfidf = dataReader.get_tfidf_artists()
    _ = dataReader.get_tfidf_album()

    recommender_list = [
        # Random,
        # TopPop,
        # RP3betaRecommender,
        # ItemKNNCBFRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # SLIM_BPR_Cython,
        UserKNNCBRecommender,
        # SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    weights = [
        1,

    ]

    # content topk = 60 e shrink = 5
    # topK = [100, 150, 50, 150, 150]
    topk = [200]

    shrinks = [10]

    # For hybrid with weighted estimated rating
    d_weights = [[0.4, 0.03863232277574469, 0.008527738266632112, 0.2560912624445676, 0.7851755932819731,0.4112843940329439, 0.3112843940329439],
        [0.2, 0.012499871230102988, 0.020242981888115352, 0.9969708006657074, 0.9999132876156388, 0.6888103295594851, 0.4888103295594851],
        [0.2, 0.10389111810225915, 0.14839466129917822, 0.866992903043857, 0.07010619211847613, 0.5873532658846817]]

    pop = [130, 346]
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
        print(type(UCM_tfidf))
        recommender = recommender_class(URM_train, ICM, recommender_list, UCM_train=UCM_tfidf, dynamic=True,
                                        d_weights=d_weights,
                                        URM_validation=URM_validation)
        # lambda_i = 0.1
        # lambda_j = 0.01
        # old_similrity_matrix = None
        # num_factors = 165
        recommender.fit(**{
                            "topK": topk,
                        "weights": weights,
                           "shrink": shrinks,
                           "pop": pop,
                            "force_compute_sim": False,
                           'alphaP3': 0.6048420766420062,
                           'alphaRP3': 1.5890147620983466,
                           'betaRP': 0.28778362462762974},)

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

    #delete_previous_intermediate_computations()
