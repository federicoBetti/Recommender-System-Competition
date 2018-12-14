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

    # delete_previous_intermediate_computations()
    if not evaluate_algorithm:
        delete_previous_intermediate_computations()
    else:
        print("ATTENTION: old intermediate computations kept, pay attention if running with all_train")

    filename = "hybrid_new_clusterization.csv"

    dataReader = RS_Data_Loader(all_train=not evaluate_algorithm)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    UCM_tfidf = dataReader.get_tfidf_artists()
    # _ = dataReader.get_tfidf_album()

    # URM_train = dataReader.get_page_rank_URM()

    recommender_list1 = [
        # Random,
        # TopPop,
        ItemKNNCBFRecommender,
        UserKNNCBRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        PureSVDRecommender
    ]
    recommender_list2 = [
        # Random,
        # TopPop,
        ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        # P3alphaRecommender,
        RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        PureSVDRecommender
    ]
    recommender_list3 = [
        # Random,
        # TopPop,
        ItemKNNCBFRecommender,
        UserKNNCBRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
        PureSVDRecommender
    ]

    d_weights_3 = [0, 0.10003298046702414, 0.7151535303946209, 0.7485507094041199]
    # For hybrid with weighted estimated rating

    # I PESI SONO RISPETTIVAMENTE DI: ItemCB, itemCF, UserCF, RP3beta, PureSVD
    # STESSO ORDINE PER SHRINK E KNN

    #'weights1': 0.06522977240989092, 'weights2': 0.18473200718460092, 'weights3': 0.48619405150100203, 'weights4': 0.404944119100937, 'weights5': 0.00024154894436601015}

    d_weights = [
        [0.45590938562950867, 0.017972928905949592, 0.23905548168035573,
         0.017005850670624212, 0.9443556793576228, 0.19081956929601618,
         0.11601757370322985, 0.11267140391070507] + [0] * len(recommender_list2) + [0] * len(recommender_list3),
        [0] * len(recommender_list1) + [0.06522977240989092, 0.18473200718460092, 0.48619405150100203,
                                        0.404944119100937, 0.00024154894436601015]
        + [0] * len(recommender_list3),
        [0] * len(recommender_list1) + [0] * len(recommender_list2) + [0, 0, 0.10003298046702414, 0.7151535303946209, 0,
                                                                       0.7485507094041199, 0,
                                                                       0.3074867937491681]
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
        onPop = True
        # On pop it used to choose if have dynamic weights for
        recommender = recommender_class(URM_train, ICM, recommender_list, dynamic=True,
                                        d_weights=d_weights, UCM_train=UCM_tfidf,
                                        URM_validation=URM_validation, onPop=onPop)

        lambda_i = 0.1
        lambda_j = 0.05
        old_similrity_matrix = None
        num_factors = 165
        l1_ratio = 1e-06

        # Variabili secondo intervallo
        alphaRP3_2 = 0.9223827655310622
        betaRP3_2 =  0.2213306613226453
        num_factors_2 =  391


        recommender.fit(**{
            "topK": [0, 0, 0, 0, 0] + [21, 220, 160, 70, -1] + [0, 0, 0, 0, 0],
            "shrink": [0, 0, 0, 0, 0] + [75, 1, 150, -1, -1] + [0, 0, 0, 0, 0],
            "pop": [130, 346],
            "weights": [1, 1, 1, 1, 1, 1, 1, 1],
            "force_compute_sim": True,
            "feature_weighting_index": 0,
            "old_similarity_matrix": old_similrity_matrix,
            "epochs": 50, "lambda_i": lambda_i,
            "lambda_j": lambda_j,
            "num_factors": num_factors,
            'alphaP3': 1.160296393373262,
            'alphaRP3': 0.49774549098196397,
            'betaRP': 0.2333486973947896,
            'l1_ratio': l1_ratio,
            "weights_to_dweights": -1})

        print("Starting Evaluations...")
        # to indicate if plotting for lenght or for pop

        results_run, results_run_string, target_recommendations = evaluator.evaluateRecommender(recommender,
                                                                                                plot_stats=True,
                                                                                                onPop=onPop)

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
