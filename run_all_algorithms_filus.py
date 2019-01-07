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
    delete_old_computations = False
    slim_after_hybrid = False

    if not evaluate_algorithm:
        delete_previous_intermediate_computations()
    else:
        print("ATTENTION: old intermediate computations kept, pay attention if running with all_train")

    filename = "URM_tfidf.csv"

    dataReader = RS_Data_Loader(all_train=not evaluate_algorithm)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    UCM_tfidf = dataReader.get_tfidf_artists()
    # _ = dataReader.get_tfidf_album()

    recommender_list = [
        # Random,
        # TopPop,
        # ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        # PureSVDRecommender
    ]

    weights = [
        1,
        # 1
    ]

    # content topk = 60 e shrink = 5
    # topK = [100, 150]
    # topk = [60, 100, 150, 56, 146, 50, -1, -1]

    # shrinks = [5, 10, 50, 10, -1, -1, -1, -1]
    # shrinks = [5, 50]

    # For hybrid with weighted estimated rating
    d_weights = [
        [0.45590938562950867, 0.017972928905949592, 0.23905548168035573, 0.017005850670624212, 0.9443556793576228,
         0.19081956929601618, 0.11601757370322985, 0.11267140391070507],
        [0.973259052781316, 0.037386979507335605, 0.8477517414017691, 0.33288193455193427, 0.9696801027638645,
         0.4723616073494711, 0.5939341460905799, 0.4188403112229081],
        [0.28230055912596863, 0.16247739973707465, 0.805610621042323, 0.8154550481439302, 0.9548692423411846,
         0.6687733529933616, 0.7785004291094528, 0.9255473647188621]]

    d_best = [[0.4, 0.03863232277574469, 0.008527738266632112, 0.2560912624445676, 0.7851755932819731,
               0.4112843940329439],
              [0.2, 0.012499871230102988, 0.020242981888115352, 0.9969708006657074, 0.9999132876156388,
               0.6888103295594851],
              [0.2, 0.10389111810225915, 0.14839466129917822, 0.866992903043857, 0.07010619211847613,
               0.5873532658846817]]

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
        Our optimal run
        '''
        onPop = True
        # On pop it used to choose if have dynamic weights for
        recommender = recommender_class(URM_train, ICM, recommender_list, UCM_train=UCM_tfidf, dynamic=False,
                                        d_weights=d_weights,
                                        URM_validation=URM_validation, onPop=onPop)

        lambda_i = 0.1
        lambda_j = 0.05
        old_similrity_matrix = None
        num_factors = 165
        recommender.fit({
            "topK": [10, 220, 150, 160, 61, 236, 40],
            "shrink": [180, 0, 15, 2, -1, -1, -1],
            "pop": [130, 346],
            "weights": [1, 1, 1, 1],
            "force_compute_sim": False,
            "feature_weighting_index": 0,
            "old_similarity_matrix": old_similrity_matrix,
            "epochs": 50,
            'alphaP3': [0.5203791059230995],
            'alphaRP3': [0.3855771543086173],
            'betaRP': [0.5217815631262526],
            'l1_ratio': 2.726530612244898e-05,
            "weights_to_dweights": -1,
            "tfidf": [True, False]})

        print("Starting Evaluations...")
        # to indicate if plotting for lenght or for pop

        results_run, results_run_string, target_recommendations = evaluator.evaluateRecommender(recommender,
                                                                                                plot_stats=True,
                                                                                                onPop=onPop)

        # print('max value in similarty slim', str(recommender.recommender_list[0].W_sparse.max()))
        # print('min value in similarity slim: ', str(recommender.recommender_list[0].W_sparse.min()))
        # print('Shape SLIM similarity: ', str(recommender.recommender_list[0].W_sparse.shape))
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
