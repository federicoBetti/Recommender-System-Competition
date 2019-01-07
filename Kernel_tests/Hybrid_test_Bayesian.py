import sys

import traceback, os
from enum import Enum
from functools import partial
from random import shuffle

import gc

from sklearn.linear_model import ElasticNet

kernel = False

import datetime
import pandas as pd
import numpy as np
import time
import pickle
from scipy.sparse import csr_matrix, csc_matrix

if kernel:
    from bayes_opt import BayesianOptimization
else:
    from ParameterTuning.BayesianOptimization_master.bayes_opt.bayesian_optimization import BayesianOptimization

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import normalize
from scipy import sparse as sps

'''
For Bayesian Search
'''


class DictionaryKeys(Enum):
    CONSTRUCTOR_POSITIONAL_ARGS = 'constructor_positional_args'
    CONSTRUCTOR_KEYWORD_ARGS = 'constructor_keyword_args'
    FIT_POSITIONAL_ARGS = 'fit_positional_args'
    FIT_KEYWORD_ARGS = 'fit_keyword_args'
    FIT_RANGE_KEYWORD_ARGS = 'fit_range_keyword_args'
    LOG_LABEL = 'log_label'


def from_fit_params_to_saved_params_function_default(recommender, paramether_dictionary):
    paramether_dictionary = paramether_dictionary.copy()

    # Attributes that might be determined through early stopping
    # Name in param_dictionary: name in object
    attributes_to_clone = {"epochs": 'epochs_best', "max_epochs": 'epochs_best'}

    for external_attribute_name in attributes_to_clone:

        recommender_attribute_name = attributes_to_clone[external_attribute_name]

        if hasattr(recommender, recommender_attribute_name):
            paramether_dictionary[external_attribute_name] = getattr(recommender, recommender_attribute_name)

    return paramether_dictionary


class EvaluatorWrapper(object):
    def __init__(self, evaluator_object):
        self.evaluator_object = evaluator_object

    def evaluateRecommender(self, recommender_object, paramether_dictionary=None):
        return self.evaluator_object.evaluateRecommender(recommender_object)


class AbstractClassSearch(object):
    ALGORITHM_NAME = "AbstractClassSearch"

    def __init__(self, recommender_class,
                 evaluator_validation=None, evaluator_test=None,
                 from_fit_params_to_saved_params_function=None):

        super(AbstractClassSearch, self).__init__()

        self.recommender_class = recommender_class

        self.results_test_best = {}
        self.paramether_dictionary_best = {}

        if evaluator_validation is None:
            raise ValueError("AbstractClassSearch: evaluator_validation must be provided")
        else:
            self.evaluator_validation = evaluator_validation

        if evaluator_test is None:
            self.evaluator_test = None
        else:
            self.evaluator_test = evaluator_test

        if from_fit_params_to_saved_params_function is None:
            self.from_fit_params_to_saved_params_function = from_fit_params_to_saved_params_function_default
        else:
            self.from_fit_params_to_saved_params_function = from_fit_params_to_saved_params_function

    def search(self, dictionary_input, metric="map", logFile=None, parallelPoolSize=2, parallelize=True):
        raise NotImplementedError("Function search not implementated for this class")

    def evaluate_on_test(self):

        # Create an object of the same class of the imput
        # Passing the paramether as a dictionary
        recommender = self.recommender_class(*self.dictionary_input[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                             **self.dictionary_input[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])

        # if self.save_model != "no":
        #     recommender.loadModel(self.output_root_path, file_name="_best_model")
        #
        # else:
        #     recommender.fit(*self.dictionary_input[DictionaryKeys.FIT_POSITIONAL_ARGS],
        #                     **self.dictionary_input[DictionaryKeys.FIT_KEYWORD_ARGS],
        #                     **self.best_solution_parameters)

        # I must do that with hybrid because since I haven't saved the model due to lot of inner recommender,
        # I can neither load it here, so I must fit it again
        recommender.fit(*self.dictionary_input[DictionaryKeys.FIT_POSITIONAL_ARGS],
                        **self.dictionary_input[DictionaryKeys.FIT_KEYWORD_ARGS],
                        **self.best_solution_parameters)

        result_dict, result_string, _ = self.evaluator_test.evaluateRecommender(recommender,
                                                                                self.best_solution_parameters)
        result_dict = result_dict[list(result_dict.keys())[0]]

        return result_dict


class BayesianSearch(AbstractClassSearch):
    ALGORITHM_NAME = "BayesianSearch"

    """
    This class applies Bayesian parameter tuning using this package:
    https://github.com/fmfn/BayesianOptimization

    pip install bayesian-optimization
    """

    def __init__(self, recommender_class, evaluator_validation=None, evaluator_test=None):

        super(BayesianSearch, self).__init__(recommender_class,
                                             evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

    def search(self, dictionary, metric="MAP", init_points=8, n_cases=20, output_root_path=None, parallelPoolSize=2,
               parallelize=True,
               save_model="best"):
        '''

        :param dictionary:
        :param metric: metric to optimize
        :param init_points: number of initial points to test before going down in the closes minimum
        :param n_cases: number of cases starting from the best init_point to find the minimum
        :param output_root_path:
        :param parallelPoolSize:
        :param parallelize:
        :param save_model:
        :return:
        '''

        # Associate the params that will be returned by BayesianOpt object to those you want to save
        # E.g. with early stopping you know which is the optimal number of epochs only afterwards
        # but you might want to save it as well
        self.from_fit_params_to_saved_params = {}

        self.dictionary_input = dictionary.copy()

        # in this variable hyperparamethers_range_dictionary there is the dictionary with all params to test
        hyperparamethers_range_dictionary = dictionary[DictionaryKeys.FIT_RANGE_KEYWORD_ARGS].copy()

        self.output_root_path = output_root_path
        if self.output_root_path is not None:
            self.logFile = self.output_root_path + "_BayesianSearch.txt"
        self.save_model = save_model
        self.model_counter = 0

        self.categorical_mapper_dict_case_to_index = {}
        self.categorical_mapper_dict_index_to_case = {}

        # Transform range element in a list of two elements: min, max
        for key in hyperparamethers_range_dictionary.keys():

            # Get the extremes for every range
            current_range = hyperparamethers_range_dictionary[key]

            if type(current_range) is range:
                min_val = current_range.start
                max_val = current_range.stop

            elif type(current_range) is list:

                categorical_mapper_dict_case_to_index_current = {}
                categorical_mapper_dict_index_to_case_current = {}

                for current_single_case in current_range:
                    num_vaues = len(categorical_mapper_dict_case_to_index_current)
                    categorical_mapper_dict_case_to_index_current[current_single_case] = num_vaues
                    categorical_mapper_dict_index_to_case_current[num_vaues] = current_single_case

                num_vaues = len(categorical_mapper_dict_case_to_index_current)

                min_val = 0
                max_val = num_vaues - 1

                self.categorical_mapper_dict_case_to_index[key] = categorical_mapper_dict_case_to_index_current.copy()
                self.categorical_mapper_dict_index_to_case[key] = categorical_mapper_dict_index_to_case_current.copy()

            else:
                raise TypeError(
                    "BayesianSearch: for every parameter a range may be specified either by a 'range' object or by a list."
                    "Provided object type for parameter '{}' was '{}'".format(key, type(current_range)))

            hyperparamethers_range_dictionary[key] = [min_val, max_val]

        self.runSingleCase_partial = partial(self.runSingleCase,
                                             dictionary=dictionary,
                                             metric=metric)

        self.bayesian_optimizer = BayesianOptimization(self.runSingleCase_partial, hyperparamethers_range_dictionary)

        self.best_solution_val = None
        self.best_solution_parameters = None
        # self.best_solution_object = None

        print("Starting the Maximize function!")
        self.bayesian_optimizer.maximize(init_points=init_points, n_iter=n_cases, kappa=2)
        #
        # best_solution = self.bayesian_optimizer.res['max']
        #
        # self.best_solution_val = best_solution["max_val"]
        # self.best_solution_parameters = best_solution["max_params"].copy()
        # self.best_solution_parameters = self.parameter_bayesian_to_token(self.best_solution_parameters)
        # self.best_solution_parameters = self.from_fit_params_to_saved_params[
        #     frozenset(self.best_solution_parameters.items())]
        #
        # print("BayesianSearch: Best config is: Config {}, {} value is {:.4f}\n".format(
        #     self.best_solution_parameters, metric, self.best_solution_val))

        # return self.best_solution_parameters.copy()
        return 1

    def parameter_bayesian_to_token(self, paramether_dictionary):
        """
        The function takes the random values from BayesianSearch and transforms them in the corresponding categorical
        tokens
        :param paramether_dictionary:
        :return:
        """

        # Convert categorical values
        for key in paramether_dictionary.keys():

            if key in self.categorical_mapper_dict_index_to_case:
                float_value = paramether_dictionary[key]
                index = int(round(float_value, 0))

                categorical = self.categorical_mapper_dict_index_to_case[key][index]

                paramether_dictionary[key] = categorical

        return paramether_dictionary

    def runSingleCase(self, dictionary, metric, **paramether_dictionary_input):

        paramether_dictionary = self.parameter_bayesian_to_token(paramether_dictionary_input)

        return self.runSingleCase_param_parsed(dictionary, metric, paramether_dictionary)

    def runSingleCase_param_parsed(self, dictionary, metric, paramether_dictionary):

        if time.time() - start_time > 60 * 60 * 5:
            return -np.inf

        try:

            # Create an object of the same class of the imput
            # Passing the paramether as a dictionary
            recommender = self.recommender_class(*dictionary[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                                 **dictionary[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])

            print("BayesianSearch: Testing config: {}".format(paramether_dictionary))

            recommender.fit(*dictionary[DictionaryKeys.FIT_POSITIONAL_ARGS],
                            **dictionary[DictionaryKeys.FIT_KEYWORD_ARGS],
                            **paramether_dictionary)

            # return recommender.evaluateRecommendations(self.URM_validation, at=5, mode="sequential")
            result_dict, _, _ = self.evaluator_validation.evaluateRecommender(recommender, paramether_dictionary)
            result_dict = result_dict[list(result_dict.keys())[0]]

            paramether_dictionary_to_save = self.from_fit_params_to_saved_params_function(recommender,
                                                                                          paramether_dictionary)

            self.from_fit_params_to_saved_params[
                frozenset(paramether_dictionary.items())] = paramether_dictionary_to_save

            self.model_counter += 1

            if self.best_solution_val is None or self.best_solution_val < result_dict[metric]:

                self.write_log(
                    "BayesianSearch: New best config found. Config: {} - MAP results: {} - time: {}\n".format(
                        paramether_dictionary_to_save, result_dict[metric], datetime.datetime.now()))
                print(
                    "BayesianSearch: New best config found. Config: {} - MAP results: {} - time: {}\n".format(
                        paramether_dictionary_to_save, result_dict[metric], datetime.datetime.now()))

                self.best_solution_val = result_dict[metric]

                if self.evaluator_test is not None:
                    self.evaluate_on_test()

            else:
                self.write_log("BayesianSearch: Config is suboptimal. Config: {} - MAP results: {} - time: {}\n".format(
                    paramether_dictionary_to_save, result_dict[metric], datetime.datetime.now()))

                print("BayesianSearch: Config is suboptimal. Config: {} - MAP results: {} - time: {}\n".format(
                    paramether_dictionary_to_save, result_dict[metric], datetime.datetime.now()))
            del recommender
            return result_dict[metric]


        except Exception as e:
            print("BayesianSearch: Testing config: {} - Exception {}\n".format(paramether_dictionary, str(e)))
            traceback.print_exc()

            return - np.inf

    def write_log(self, string):
        with open(self.logFile, 'a') as the_file:
            the_file.write(string + '\n')


'''
Sequential Evaluator
'''


def roc_auc(is_relevant):
    ranks = np.arange(len(is_relevant))
    pos_ranks = ranks[is_relevant]
    neg_ranks = ranks[~is_relevant]
    auc_score = 0.0

    if len(neg_ranks) == 0:
        return 1.0

    if len(pos_ranks) > 0:
        for pos_pred in pos_ranks:
            auc_score += np.sum(pos_pred < neg_ranks, dtype=np.float32)
        auc_score /= (pos_ranks.shape[0] * neg_ranks.shape[0])

    assert 0 <= auc_score <= 1, auc_score
    return auc_score


def arhr(is_relevant):
    # average reciprocal hit-rank (ARHR)
    # pag 17
    # http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf
    # https://emunix.emich.edu/~sverdlik/COSC562/ItemBasedTopTen.pdf

    p_reciprocal = 1 / np.arange(1, len(is_relevant) + 1, 1.0, dtype=np.float64)
    arhr_score = is_relevant.dot(p_reciprocal)

    assert 0 <= arhr_score <= p_reciprocal.sum(), arhr_score
    return arhr_score


def precision(is_relevant, n_test_items):
    precision_score = np.sum(is_relevant, dtype=np.float32) / min(n_test_items, len(is_relevant))

    assert 0 <= precision_score <= 1, precision_score
    return precision_score


def recall_min_test_len(is_relevant, pos_items):
    recall_score = np.sum(is_relevant, dtype=np.float32) / min(pos_items.shape[0], len(is_relevant))

    assert 0 <= recall_score <= 1, recall_score
    return recall_score


def recall(is_relevant, pos_items):
    recall_score = np.sum(is_relevant, dtype=np.float32) / pos_items.shape[0]

    assert 0 <= recall_score <= 1, recall_score
    return recall_score


def rr(is_relevant):
    # reciprocal rank of the FIRST relevant item in the ranked list (0 if none)

    ranks = np.arange(1, len(is_relevant) + 1)[is_relevant]

    if len(ranks) > 0:
        return 1. / ranks[0]
    else:
        return 0.0


def map(is_relevant, pos_items):
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([pos_items.shape[0], is_relevant.shape[0]])

    assert 0 <= map_score <= 1, map_score
    return map_score


def ndcg(ranked_list, pos_items, relevance=None, at=None):
    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    # Creates array of length "at" with the relevance associated to the item in that position
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)

    # IDCG has all relevances to 1, up to the number of items in the test set
    ideal_dcg = dcg(np.sort(relevance)[::-1])

    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg
    # assert 0 <= ndcg_ <= 1, (rank_dcg, ideal_dcg, ndcg_)
    return ndcg_


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)


metrics = ['AUC', 'Precision' 'Recall', 'MAP', 'NDCG']


class EvaluatorMetrics(Enum):
    ROC_AUC = "ROC_AUC"
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    RECALL_TEST_LEN = "RECALL_TEST_LEN"
    MAP = "MAP"
    MRR = "MRR"
    NDCG = "NDCG"
    F1 = "F1"
    HIT_RATE = "HIT_RATE"
    ARHR = "ARHR"
    NOVELTY = "NOVELTY"
    DIVERSITY_SIMILARITY = "DIVERSITY_SIMILARITY"
    DIVERSITY_MEAN_INTER_LIST = "DIVERSITY_MEAN_INTER_LIST"
    DIVERSITY_HERFINDAHL = "DIVERSITY_HERFINDAHL"
    COVERAGE_ITEM = "COVERAGE_ITEM"
    COVERAGE_USER = "COVERAGE_USER"
    DIVERSITY_GINI = "DIVERSITY_GINI"
    SHANNON_ENTROPY = "SHANNON_ENTROPY"


def create_empty_metrics_dict():
    empty_dict = {}

    # from Base.Evaluation.ResultMetric import ResultMetric
    # empty_dict = ResultMetric()

    for metric in EvaluatorMetrics:
        empty_dict[metric.value] = 0.0

    return empty_dict


class Evaluator(object):
    """Abstract Evaluator"""

    EVALUATOR_NAME = "Evaluator_Base_Class"

    def __init__(self, URM_test_list, cutoff_list, minRatingsPerUser=1, exclude_seen=True,
                 diversity_object=None,
                 ignore_items=None,
                 ignore_users=None):

        super(Evaluator, self).__init__()

        if ignore_items is None:
            self.ignore_items_flag = False
            self.ignore_items_ID = np.array([])
        else:
            print("Ignoring {} Items".format(len(ignore_items)))
            self.ignore_items_flag = True
            self.ignore_items_ID = np.array(ignore_items)

        self.cutoff_list = cutoff_list.copy()
        self.max_cutoff = max(self.cutoff_list)

        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen

        if not isinstance(URM_test_list, list):
            self.URM_test = URM_test_list.copy()
            URM_test_list = [URM_test_list]
        else:
            raise ValueError("List of URM_test not supported")

        self.diversity_object = diversity_object

        self.n_users = URM_test_list[0].shape[0]
        self.n_items = URM_test_list[0].shape[1]

        # Prune users with an insufficient number of ratings
        # During testing CSR is faster
        self.URM_test_list = []
        usersToEvaluate_mask = np.zeros(self.n_users, dtype=np.bool)

        for URM_test in URM_test_list:
            URM_test = csr_matrix(URM_test)
            self.URM_test_list.append(URM_test)

            rows = URM_test.indptr
            numRatings = np.ediff1d(rows)
            new_mask = numRatings >= minRatingsPerUser

            usersToEvaluate_mask = np.logical_or(usersToEvaluate_mask, new_mask)

        self.usersToEvaluate = np.arange(self.n_users)[usersToEvaluate_mask]

        if ignore_users is not None:
            print("Ignoring {} Users".format(len(ignore_users)))
            self.ignore_users_ID = np.array(ignore_users)
            self.usersToEvaluate = set(self.usersToEvaluate) - set(ignore_users)
        else:
            self.ignore_users_ID = np.array([])

        self.usersToEvaluate = list(self.usersToEvaluate)

    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        raise NotImplementedError("The method evaluateRecommender not implemented for this evaluator class")

    def get_user_relevant_items(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in getting relevant items"

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id + 1]]

    def get_user_test_ratings(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in relevant items ratings"

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id + 1]]

    def get_result_string(self, results_run):

        output_str = ""

        for cutoff in results_run.keys():

            results_run_current_cutoff = results_run[cutoff]

            output_str += "CUTOFF: {} - ".format(cutoff)

            for metric in results_run_current_cutoff.keys():
                output_str += "{}: {:.7f}, ".format(metric, results_run_current_cutoff[metric])

            output_str += "\n"

        return output_str


class SequentialEvaluator(Evaluator):
    """SequentialEvaluator"""

    EVALUATOR_NAME = "SequentialEvaluator_Class"

    def __init__(self, URM_test_list, URM_train, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True,
                 diversity_object=None,
                 ignore_items=None,
                 ignore_users=None):

        self.URM_train = URM_train
        super(SequentialEvaluator, self).__init__(URM_test_list, cutoff_list,
                                                  diversity_object=diversity_object,
                                                  minRatingsPerUser=minRatingsPerUser, exclude_seen=exclude_seen,
                                                  ignore_items=ignore_items, ignore_users=ignore_users)

    def _run_evaluation_on_selected_users(self, recommender_object, usersToEvaluate, block_size=1000, plot_stats=False,
                                          onPop=True):

        to_ret = []
        start_time = time.time()
        start_time_print = time.time()

        results_dict = {}

        for cutoff in self.cutoff_list:
            results_dict[cutoff] = create_empty_metrics_dict()

        n_users_evaluated = 0

        data_stats_pop = {}
        data_stats_len = {}
        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0
        while user_batch_start < len(self.usersToEvaluate):
            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(usersToEvaluate))

            test_user_batch_array = np.array(usersToEvaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one
            # at a time
            recommended_items_batch_list = recommender_object.recommend(test_user_batch_array,
                                                                        remove_seen_flag=self.exclude_seen,
                                                                        cutoff=self.max_cutoff,
                                                                        remove_top_pop_flag=False,
                                                                        remove_CustomItems_flag=self.ignore_items_flag
                                                                        )

            # Compute recommendation quality for each user in batch
            for batch_user_index in range(len(recommended_items_batch_list)):

                user_id = test_user_batch_array[batch_user_index]
                recommended_items = recommended_items_batch_list[batch_user_index]

                # Being the URM CSR, the indices are the non-zero column indexes
                relevant_items = self.get_user_relevant_items(user_id)
                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                user_profile = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

                # added to_ret here
                to_ret.append((user_id, recommended_items[:self.max_cutoff]))
                n_users_evaluated += 1

                for cutoff in self.cutoff_list:
                    results_current_cutoff = results_dict[cutoff]

                    is_relevant_current_cutoff = is_relevant[0:cutoff]
                    recommended_items_current_cutoff = recommended_items[0:cutoff]

                    current_map = map(is_relevant_current_cutoff, relevant_items)
                    results_current_cutoff[EvaluatorMetrics.ROC_AUC.value] += roc_auc(is_relevant_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.PRECISION.value] += precision(is_relevant_current_cutoff,
                                                                                          len(relevant_items))
                    results_current_cutoff[EvaluatorMetrics.RECALL.value] += recall(is_relevant_current_cutoff,
                                                                                    relevant_items)
                    results_current_cutoff[EvaluatorMetrics.RECALL_TEST_LEN.value] += recall_min_test_len(
                        is_relevant_current_cutoff, relevant_items)
                    results_current_cutoff[EvaluatorMetrics.MAP.value] += current_map
                    results_current_cutoff[EvaluatorMetrics.MRR.value] += rr(is_relevant_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.NDCG.value] += ndcg(recommended_items_current_cutoff,
                                                                                relevant_items,
                                                                                relevance=self.get_user_test_ratings(
                                                                                    user_id), at=cutoff)
                    results_current_cutoff[EvaluatorMetrics.HIT_RATE.value] += is_relevant_current_cutoff.sum()
                    results_current_cutoff[EvaluatorMetrics.ARHR.value] += arhr(is_relevant_current_cutoff)

                    # create both data structures for plotting: lenght and popularity

        return results_dict, n_users_evaluated, to_ret

    def evaluateRecommender(self, recommender_object, plot_stats=False, onPop=True):
        """
        :param recommender_object: the trained recommenderURM_validation object, a Recommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        results_dict, n_users_evaluated, to_ret_values = self._run_evaluation_on_selected_users(recommender_object,
                                                                                                self.usersToEvaluate,
                                                                                                plot_stats=plot_stats,
                                                                                                onPop=onPop)

        if (n_users_evaluated > 0):

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                for key in results_current_cutoff.keys():
                    value = results_current_cutoff[key]

                    results_current_cutoff[key] = value / n_users_evaluated

                precision_ = results_current_cutoff[EvaluatorMetrics.PRECISION.value]
                recall_ = results_current_cutoff[EvaluatorMetrics.RECALL.value]

                if precision_ + recall_ != 0:
                    results_current_cutoff[EvaluatorMetrics.F1.value] = 2 * (precision_ * recall_) / (
                        precision_ + recall_)


        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run_string = self.get_result_string(results_dict)

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        return (results_dict, results_run_string, to_ret_values)


'''
RS Data Loader
'''


def divide_train_test(train_old, threshold=0.8):
    msk = np.random.rand(len(train_old)) < threshold
    train = train_old[msk]
    test = train_old[~msk]
    return train, test


def create_URM_matrix(train):
    row = list(train.playlist_id)
    col = list(train.track_id)
    return csr_matrix(([1] * len(row), (row, col)), shape=(50446, 20635))


def get_icm_matrix(tracks):
    tracks_arr = tracks.track_id.values
    album_arr = tracks.album_id.unique()
    artists_arr = tracks.artist_id.unique()
    feature_tracks = np.ndarray(shape=(tracks_arr.shape[0], album_arr.shape[0] + artists_arr.shape[0]))

    def create_feature(entry):
        feature_tracks[entry.track_id][entry.album_id] = 1
        feature_tracks[entry.track_id][album_arr.shape[0] + entry.artist_id] = 0.8

    tracks.apply(create_feature, axis=1)
    to_ret = csr_matrix(feature_tracks)
    del feature_tracks
    return to_ret


class RS_Data_Loader(object):
    def __init__(self, split_train_test_validation_quota=[0.8, 0.0, 0.2], distr_split=True, top10k=False,
                 all_train=False):
        super(RS_Data_Loader, self).__init__()

        if abs(sum(split_train_test_validation_quota) - 1.0) > 0.001 or len(split_train_test_validation_quota) != 3:
            raise ValueError(
                "RS_Data_Loader: splitTrainTestValidation must be a probability distribution over Train, Test and Validation")

        print("RS_Data_Loader: loading data...")

        self.all_train = all_train
        self.top10k = top10k
        self.distr_split = distr_split
        if kernel:
            self.train = pd.read_csv(os.path.join("../input", "train.csv"))
            self.tracks = pd.read_csv(os.path.join("../input", "tracks.csv"))
            self.target_playlist = pd.read_csv(os.path.join("../input", "target_playlists.csv"))
        else:
            self.train = pd.read_csv(os.path.join("../Dataset", "train.csv"))
            self.tracks = pd.read_csv(os.path.join("../Dataset", "tracks.csv"))
            self.target_playlist = pd.read_csv(os.path.join("../Dataset", "target_playlists.csv"))
        self.ICM = None

        train, test = divide_train_test(self.train, threshold=0.85)

        self.URM_train = create_URM_matrix(train)
        self.URM_test = create_URM_matrix(test)
        self.URM_validation = self.URM_test

        print("RS_Data_Loader: loading complete")

    def get_URM_train(self):
        return self.URM_train

    def get_URM_test(self):
        return self.URM_test

    def get_URM_validation(self):
        return self.URM_validation

    def get_target_playlist(self):
        return self.target_playlist

    def get_traks(self):
        return self.tracks

    def get_tfidf_artists(self):
        return self.UCB_tfidf_artists

    def get_tfidf_album(self):
        return self.UCB_tfidf_album

    def create_complete_test(self):
        row = 50446
        col = 20635
        return csr_matrix(([1] * row, (range(row), [0] * row)), shape=(row, col))

    def get_ICM(self):
        if self.ICM is None:
            self.ICM = get_icm_matrix(self.tracks)
        return self.ICM


'''
RP3beta, this is the part related to the single recommender that one should change
'''


def playlist_popularity(playlist_songs, pop_dict):
    pop = 0
    count = 0
    for track in playlist_songs:
        pop += pop_dict[track]
        count += 1

    if count == 0:
        return 0

    return pop / count


def lenght_playlist(playlist_songs):
    return len(playlist_songs)


def get_tfidf(matrix):
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(matrix)
    if isinstance(tfidf, csr_matrix):
        return tfidf
    else:
        return csr_matrix(tfidf.toarray())


class Compute_Similarity_Python:
    def __init__(self, dataMatrix, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine", row_weights=None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions

        """
        """
        Asymmetric Cosine as described in:
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.

        """

        super(Compute_Similarity_Python, self).__init__()

        self.TopK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for paramether 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))

        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_columns, self.n_columns))

        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Col_weights has {} columns, dataMatrix has {}.".format(len(row_weights),
                                                                                         dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T

    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csr')

        interactionsPerRow = np.diff(self.dataMatrix.indptr)

        nonzeroRows = interactionsPerRow > 0
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sumPerRow)
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]

        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row = 0

        blockSize = 1000

        while end_row < self.n_rows:
            end_row = min(self.n_rows, end_row + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize

    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]

        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col = 0

        blockSize = 1000

        while end_col < self.n_columns:
            end_col = min(self.n_columns, end_col + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize

    def useOnlyBooleanInteractions(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos = 0

        blockSize = 1000

        while end_pos < len(self.dataMatrix.data):
            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos - start_pos)

            start_pos += blockSize

    def compute_similarity(self, start_col=None, end_col=None, block_size=100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.useOnlyBooleanInteractions()

        # We explore the matrix column-wise
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        # Compute sum of squared values to be used in normalization
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sumOfSquared = np.sqrt(sumOfSquared)

        if self.asymmetric_cosine:
            sumOfSquared_to_1_minus_alpha = np.power(sumOfSquared, 2 * (1 - self.asymmetric_alpha))
            sumOfSquared_to_alpha = np.power(sumOfSquared, 2 * self.asymmetric_alpha)

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col > 0 and start_col < self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col > start_col_local and end_col < self.n_columns:
            end_col_local = end_col

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            # Add previous block size
            processedItems += this_block_size

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            if time.time() - start_time_print_batch >= 30 or end_col_block == end_col_local:
                columnPerSec = processedItems / (time.time() - start_time)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec,
                                    (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            if self.use_row_weights:
                # item_data = np.multiply(item_data, self.row_weights)
                # item_data = item_data.T.dot(self.row_weights_diag).T
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities
                this_block_weights = self.dataMatrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]

                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:

                    if self.asymmetric_cosine:
                        denominator = sumOfSquared_to_alpha[
                                          columnIndex] * sumOfSquared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)


                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + \
                                  (sumOfSquared[columnIndex] - this_column_weights) * self.tversky_alpha + \
                                  (sumOfSquared - this_column_weights) * self.tversky_beta + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # this_column_weights = this_column_weights.toarray().ravel()

                if self.TopK == 0:
                    self.W_dense[:, columnIndex] = this_column_weights

                else:
                    # Sort indices and select TopK
                    # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                    # - Partition the data to extract the set of relevant items
                    # - Sort only the relevant items
                    # - Get the original item index
                    relevant_items_partition = (-this_column_weights).argpartition(self.TopK - 1)[0:self.TopK]
                    relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                    top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                    # Incrementally build sparse matrix, do not add zeros
                    notZerosMask = this_column_weights[top_k_idx] != 0.0
                    numNotZeros = np.sum(notZerosMask)

                    values.extend(this_column_weights[top_k_idx][notZerosMask])
                    rows.extend(top_k_idx[notZerosMask])
                    cols.extend(np.ones(numNotZeros) * columnIndex)

            start_col_block += block_size

        # End while on columns


        if self.TopK == 0:
            return self.W_dense

        else:

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                      shape=(self.n_columns, self.n_columns),
                                      dtype=np.float32)

            return W_sparse


class SimilarityFunction(Enum):
    COSINE = "cosine"
    PEARSON = "pearson"
    JACCARD = "jaccard"
    TANIMOTO = "tanimoto"
    ADJUSTED_COSINE = "adjusted"
    EUCLIDEAN = "euclidean"


class Compute_Similarity:
    def __init__(self, dataMatrix, use_implementation="density", similarity=None, **args):
        """
        Interface object that will call the appropriate similarity implementation
        :param dataMatrix:
        :param use_implementation:      "density" will choose the most efficient implementation automatically
                                        "cython" will use the cython implementation, if available. Most efficient for sparse matrix
                                        "python" will use the python implementation. Most efficent for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        """

        self.dense = False

        if similarity == "euclidean":
            # This is only available here
            # self.compute_similarity_object = Compute_Similarity_Euclidean(dataMatrix, **args)
            a = 1
        else:

            if similarity is not None:
                args["similarity"] = similarity

            if use_implementation == "density":

                if isinstance(dataMatrix, np.ndarray):
                    self.dense = True

                elif isinstance(dataMatrix, sps.spmatrix):
                    shape = dataMatrix.shape

                    num_cells = shape[0] * shape[1]

                    sparsity = dataMatrix.nnz / num_cells

                    self.dense = sparsity > 0.5

                else:
                    print("Compute_Similarity: matrix type not recognized, calling default...")
                    use_implementation = "python"

                if self.dense:
                    print("Compute_Similarity: detected dense matrix")
                    use_implementation = "python"
                else:
                    use_implementation = "cython"

            if use_implementation == "cython":

                try:
                    from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
                    self.compute_similarity_object = Compute_Similarity_Cython(dataMatrix, **args)

                except ImportError:
                    print("Unable to load Cython Compute_Similarity, reverting to Python")
                    self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)


            elif use_implementation == "python":
                self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)

            else:

                raise ValueError("Compute_Similarity: value for argument 'use_implementation' not recognized")

    def compute_similarity(self, **args):

        return self.compute_similarity_object.compute_similarity(**args)


class Recommender(object):
    """Abstract Recommender"""

    RECOMMENDER_NAME = "Recommender_Base_Class"

    def __init__(self):

        super(Recommender, self).__init__()

        self.URM_train = None
        self.sparse_weights = True
        self.normalize = False

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

    def fit(self):
        pass

    def change_weights(self, level, pop):
        pass

    def get_URM_train(self):
        return self.URM_train.copy()

    def set_items_to_ignore(self, items_to_ignore):

        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch

    def _remove_CustomItems_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch

    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

    def compute_item_score(self, user_id):
        raise NotImplementedError(
            "Recommender: compute_item_score not assigned for current recommender, unable to compute prediction scores")

    def recommend(self, user_id_array, dict_pop=None, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self.compute_item_score(user_id_array)

        # if self.normalize:
        #     # normalization will keep the scores in the same range
        #     # of value of the ratings in dataset
        #     user_profile = self.URM_train[user_id]
        #
        #     rated = user_profile.copy()
        #     rated.data = np.ones_like(rated.data)
        #     if self.sparse_weights:
        #         den = rated.dot(self.W_sparse).toarray().ravel()
        #     else:
        #         den = rated.dot(self.W).ravel()
        #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
        #     scores /= den


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]
            a = scores_batch[user_index, :]
            if remove_seen_flag:
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
                # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
                # ranking = relevant_items_partition[relevant_items_partition_sorting]
                #
                # ranking_list.append(ranking)

        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_CustomItems_flag:
            scores_batch = self._remove_CustomItems_on_scores(scores_batch)

        # scores_batch = np.arange(0,3260).reshape((1, -1))
        # scores_batch = np.repeat(scores_batch, 1000, axis = 0)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[
            np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking_list

    def evaluateRecommendations(self, URM_test, at=5, minRatingsPerUser=1, exclude_seen=True,
                                filterCustomItems=np.array([], dtype=np.int),
                                filterCustomUsers=np.array([], dtype=np.int)):
        """
        Speed info:
        - Sparse weighgs: batch mode is 2x faster than sequential
        - Dense weighgts: batch and sequential speed are equivalent


        :param URM_test:            URM to be used for testing
        :param at: 5                    Length of the recommended items
        :param minRatingsPerUser: 1     Users with less than this number of interactions will not be evaluated
        :param exclude_seen: True       Whether to remove already seen items from the recommended items

        :param mode: 'sequential', 'parallel', 'batch'
        :param filterTopPop: False or decimal number        Percentage of items to be removed from recommended list and testing interactions
        :param filterCustomItems: Array, default empty           Items ID to NOT take into account when recommending
        :param filterCustomUsers: Array, default empty           Users ID to NOT take into account when recommending
        :return:
        """

        import warnings

        warnings.warn("DEPRECATED! Use Base.Evaluation.SequentialEvaluator.evaluateRecommendations()",
                      DeprecationWarning)

        from Base.Evaluation.Evaluator import SequentialEvaluator

        evaluator = SequentialEvaluator(URM_test, [at], exclude_seen=exclude_seen,
                                        minRatingsPerUser=minRatingsPerUser,
                                        ignore_items=filterCustomItems, ignore_users=filterCustomUsers)

        results_run, results_run_string = evaluator.evaluateRecommender(self)

        results_run = results_run[at]

        results_run_lowercase = {}

        for key in results_run.keys():
            results_run_lowercase[key.lower()] = results_run[key]

        return results_run_lowercase


class SimilarityMatrixRecommender(object):
    """
    This class refers to a Recommender KNN which uses a similarity matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """

    def __init__(self):
        super(SimilarityMatrixRecommender, self).__init__()

        self.sparse_weights = True

        self.compute_item_score = self.compute_score_item_based

    def compute_score_item_based(self, user_id):

        if self.sparse_weights:
            user_profile = self.URM_train[user_id]

            to_ret = user_profile.dot(self.W_sparse).toarray()
            return to_ret

        else:

            assert False

            user_profile = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            user_ratings = self.URM_train.data[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

            relevant_weights = self.W[user_profile]
            return relevant_weights.T.dot(user_ratings)

    def compute_score_user_based(self, user_id):

        if self.sparse_weights:

            to_ret = self.W_sparse[user_id].dot(self.URM_train).toarray()
            return to_ret

        else:
            # Numpy dot does not recognize sparse matrices, so we must
            # invoke the dot function on the sparse one
            return self.URM_train.T.dot(self.W[user_id])


def similarityMatrixTopK(item_weights, forceSparseOutput=True, k=100, verbose=False, inplace=True):
    """
    The function selects the TopK most similar elements, column-wise

    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    if not sparse_weights:

        idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

        if inplace:
            W = item_weights
        else:
            W = item_weights.copy()

        # index of the items that don't belong to the top-k similar items of each column
        not_top_k = idx_sorted[:-k, :]
        # use numpy fancy indexing to zero-out the values in sim without using a for loop
        W[not_top_k, np.arange(nitems)] = 0.0

        if forceSparseOutput:
            W_sparse = csr_matrix(W, shape=(nitems, nitems))

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

        if verbose:
            print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W

    else:
        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        for item_idx in range(nitems):
            cols_indptr.append(len(data))

            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx + 1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

            non_zero_data = column_data != 0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-k:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        cols_indptr.append(len(data))

        # During testing CSR is faster
        W_sparse = csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
        W_sparse = W_sparse.tocsr()

        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


class RP3betaRecommender(SimilarityMatrixRecommender, Recommender):
    """ RP3beta recommender """

    RECOMMENDER_NAME = "RP3betaRecommender"

    def __init__(self, URM_train):
        super(RP3betaRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train, format='csr', dtype=np.float32)
        self.sparse_weights = True

    def __str__(self):
        return "RP3beta(alpha={}, beta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(
            self.alpha,
            self.beta, self.min_rating, self.topK,
            self.implicit, self.normalize_similarity)

    def fit(self, alpha=1., beta=0.6, min_rating=0, topK=100, implicit=True, normalize_similarity=False,
            force_compute_sim=True):

        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("RP3betaMatrix.pkl"), 'rb') as handle:
                    (topK_new, W_sparse_new) = pickle.load(handle)
            except FileNotFoundError:
                print("File {} not found".format(os.path.join("IntermediateComputations", "RP3betaMatrix.pkl")))
                found = False

            if found and self.topK == topK_new:
                self.W_sparse = W_sparse_new
                print("Saved RP3beta Similarity Matrix Used!")
                return

        # if X.dtype != np.float32:
        #     print("RP3beta fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        # Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM_train.shape[1])

        nonZeroMask = X_bool_sum != 0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W = normalize(self.W, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W, forceSparseOutput=True, k=self.topK)
            self.sparse_weights = True

        with open(os.path.join("RP3betaMatrix.pkl"), 'wb') as handle:
            pickle.dump((self.topK, self.W_sparse), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("RP3beta similarity matrix saved")


def okapi_BM_25(dataMatrix, K1=1.2, B=0.75):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :param K1:
    :param B:
    :return:
    """

    assert B > 0 and B < 1, "okapi_BM_25: B must be in (0,1)"
    assert K1 > 0, "okapi_BM_25: K1 must be > 0"

    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)

    dataMatrix = sps.coo_matrix(dataMatrix)

    N = float(dataMatrix.shape[0])
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # calculate length_norm per document
    row_sums = np.ravel(dataMatrix.sum(axis=1))

    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    dataMatrix.data = dataMatrix.data * (K1 + 1.0) / (K1 * length_norm[dataMatrix.row] + dataMatrix.data) * idf[
        dataMatrix.col]

    return dataMatrix.tocsr()


def TF_IDF(dataMatrix):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :return:
    """

    # TFIDF each row of a sparse amtrix
    dataMatrix = sps.coo_matrix(dataMatrix)
    N = float(dataMatrix.shape[0])

    # calculate IDF
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # apply TF-IDF adjustment
    dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]

    return dataMatrix.tocsr()


class ItemKNNCBFRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCBFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, ICM, URM_train, sparse_weights=True):
        super(ItemKNNCBFRecommender, self).__init__()

        self.ICM = ICM.copy()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights
        self.W_sparse = None

    def __str__(self):
        return "Item Content Based (tokK={}, shrink={}, feature_weigthing_index={}".format(
            self.topK, self.shrink, self.feature_weighting_index)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, force_compute_sim=True,
            feature_weighting="none", feature_weighting_index=0, **similarity_args):

        self.feature_weighting_index = feature_weighting_index
        feature_weighting = self.FEATURE_WEIGHTING_VALUES[feature_weighting_index]
        self.topK = topK
        self.shrink = shrink

        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("ContentBFMatrix.pkl"), 'rb') as handle:
                    (topK_new, shrink_new, W_sparse_new) = pickle.load(handle)
            except FileNotFoundError:
                print("File {} not found".format(os.path.join("IntermediateComputations", "ContentBFMatrix.pkl")))
                found = False

            if found and self.topK == topK_new and self.shrink == shrink_new:
                self.W_sparse = W_sparse_new
                print("Saved CBF Similarity Matrix Used!")
                return

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)

        elif feature_weighting == "TF-IDF":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)

        similarity = Compute_Similarity(self.ICM.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()

            with open(os.path.join("ContentBFMatrix.pkl"), 'wb') as handle:
                pickle.dump((self.topK, self.shrink, self.W_sparse), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("CBF similarity matrix saved")
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()


class ItemKNNCFRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(ItemKNNCFRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

        self.W_sparse = None
        self.topK = None
        self.shrink = None
        self.tfidf = None

    def __str__(self):
        return "Item Collaborative Filterng (tokK={}, shrink={}, tfidf={}, normalize={}".format(
            self.topK, self.shrink, self.tfidf, self.normalize)

    def fit(self, topK=350, shrink=10, similarity='cosine', normalize=True, force_compute_sim=True, tfidf=True,
            **similarity_args):

        self.topK = topK
        self.shrink = shrink
        self.tfidf = tfidf

        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("ItemCFMatrix_tfidf_{}.pkl".format(str(self.tfidf))),
                          'rb') as handle:
                    (topK_new, shrink_new, W_sparse_new) = pickle.load(handle)
            except FileNotFoundError:
                print("File {} not found".format(
                    os.path.join("IntermediateComputations", "ItemCFMatrix_tfidf_{}.pkl".format(str(self.tfidf)))))
                found = False

            if found and self.topK == topK_new and self.shrink == shrink_new:
                self.W_sparse = W_sparse_new
                print("Saved Item CF Similarity Matrix Used!")
                return

        if tfidf:
            sim_matrix_pre = get_tfidf(self.URM_train)
        else:
            sim_matrix_pre = self.URM_train

        similarity = Compute_Similarity(sim_matrix_pre, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)
        print('Similarity item based CF computed')

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
            with open(os.path.join("ItemCFMatrix_tfidf_{}.pkl".format(str(self.tfidf))),
                      'wb') as handle:
                pickle.dump((self.topK, self.shrink, self.W_sparse), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Item CF similarity matrix saved")
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()


class UserKNNCFRecommender(SimilarityMatrixRecommender, Recommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(UserKNNCFRecommender, self).__init__()

        # Not sure if CSR here is faster
        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

        self.compute_item_score = self.compute_score_user_based

        self.W_sparse = None

    def __str__(self):
        return "User Collaborative Filterng (tokK={}, shrink={}, tfidf={}, normalize={}".format(
            self.topK, self.shrink, self.tfidf, self.normalize)

    def fit(self, topK=350, shrink=10, similarity='cosine', normalize=True, force_compute_sim=True, tfidf=True,
            **similarity_args):

        self.topK = topK
        self.shrink = shrink
        self.tfidf = tfidf

        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("UserCFSimMatrix.pkl"), 'rb') as handle:
                    (topK_new, shrink_new, W_sparse_new) = pickle.load(handle)
            except FileNotFoundError:
                found = False

            if found and self.topK == topK_new and self.shrink == shrink_new:
                self.W_sparse = W_sparse_new
                print("Saved User CF Similarity Matrix Used!")
                return

        if tfidf:
            sim_matrix_pre = get_tfidf(self.URM_train)
        else:
            sim_matrix_pre = self.URM_train
        similarity = Compute_Similarity(sim_matrix_pre.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()

            with open(os.path.join("UserCFSimMatrix.pkl"), 'wb') as handle:
                pickle.dump((self.topK, self.shrink, self.W_sparse), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()


class P3alphaRecommender(SimilarityMatrixRecommender, Recommender):
    """ P3alpha recommender """

    RECOMMENDER_NAME = "P3alphaRecommender"

    def __init__(self, URM_train):
        super(P3alphaRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train, format='csr', dtype=np.float32)
        self.sparse_weights = True

    def __str__(self):
        return "P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                                        self.min_rating,
                                                                                                        self.topK,
                                                                                                        self.implicit,
                                                                                                        self.normalize_similarity)

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=True, normalize_similarity=False, force_compute_sim=True):

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("P3alphaMatrix.pkl"), 'rb') as handle:
                    (topK_new, W_sparse_new) = pickle.load(handle)
            except FileNotFoundError:
                print("File {} not found".format(os.path.join("IntermediateComputations", "P3alphaMatrix.pkl")))
                found = False

            if found and self.topK == topK_new:
                self.W_sparse = W_sparse_new
                print("Saved P3alpha Similarity Matrix Used!")
                return
        #
        # if X.dtype != np.float32:
        #     print("P3ALPHA fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        # Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, forceSparseOutput=True, k=self.topK)
            self.sparse_weights = True

        with open(os.path.join("P3alphaMatrix.pkl"), 'wb') as handle:
            pickle.dump((self.topK, self.W_sparse), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("P3alpha similarity matrix saved")


class HybridRecommender(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, ICM, recommender_list, UCM_train=None, dynamic=False, d_weights=None, weights=None,
                 URM_validation=None, sparse_weights=True, onPop=True, moreHybrids=False):
        super(Recommender, self).__init__()

        # CSR is faster during evaluation
        self.pop = None
        self.UCM_train = UCM_train
        self.URM_train = check_matrix(URM_train, 'csr')
        self.URM_validation = URM_validation
        self.dynamic = dynamic
        self.d_weights = d_weights
        self.dataset = None
        self.onPop = onPop
        self.moreHybrids = moreHybrids

        # with open(os.path.join("Dataset", "Cluster_0_dict_Kmeans_3.pkl"), 'rb') as handle:
        #     self.cluster_0 = pickle.load(handle)
        # with open(os.path.join("Dataset", "Cluster_1_dict_Kmeans_3.pkl"), 'rb') as handle:
        #     self.cluster_1 = pickle.load(handle)
        # with open(os.path.join("Dataset", "Cluster_2_dict_Kmeans_3.pkl"), 'rb') as handle:
        #     self.cluster_2 = pickle.load(handle)

        self.sparse_weights = sparse_weights

        self.recommender_list = []
        self.weights = weights

        self.normalize = None
        self.topK = None
        self.shrink = None

        self.recommender_number = len(recommender_list)
        if self.d_weights is None:
            # 3 because we have divided in 3 intervals
            self.d_weights = [[0] * self.recommender_number, [0] * self.recommender_number,
                              [0] * self.recommender_number]

        for recommender in recommender_list:
            if recommender is ItemKNNCBFRecommender:
                self.recommender_list.append(recommender(ICM, URM_train))

            elif recommender.__class__ in [HybridRecommender]:
                self.recommender_list.append(recommender)

            else:  # UserCF, ItemCF, ItemCBF, P3alpha, RP3beta
                self.recommender_list.append(recommender(URM_train))

    def fit(self, topK=None, shrink=None, weights=None, pop=None, weights1=None, weights2=None, weights3=None,
            weights4=None,
            weights5=None, weights6=None, weights7=None, weights8=None, pop1=None, pop2=None, similarity='cosine',
            normalize=True,
            old_similarity_matrix=None, epochs=1, top1=None, shrink1=None,
            force_compute_sim=False, weights_to_dweights=-1, **similarity_args):

        if topK is None:  # IT MEANS THAT I'M TESTING ONE RECOMMENDER ON A SPECIIFC INTERVAL
            topK = [top1]
            shrink = [shrink1]

        if self.weights is None:
            if weights1 is not None:
                weights = [weights1, weights2, weights3, weights4, weights5, weights6, weights7, weights8]
                weights = [x for x in weights if x is not None]
            self.weights = weights

        if self.pop is None:
            if pop is None:
                pop = [pop1, pop2]
                pop = [x for x in pop if x is not None]
            self.pop = pop

        if weights_to_dweights != -1:
            self.d_weights[weights_to_dweights] = self.weights

        assert self.weights is not None, "Weights Are None!"

        assert len(self.recommender_list) == len(
            self.weights), "Weights: {} and recommender list: {} have different lenghts".format(len(self.weights), len(
            self.recommender_list))

        assert len(topK) == len(shrink) == len(self.recommender_list), "Knns, Shrinks and recommender list have " \
                                                                       "different lenghts "

        self.normalize = normalize
        self.topK = topK
        self.shrink = shrink

        self.gradients = [0] * self.recommender_number
        self.MAE = 0
        p3counter = 0
        rp3bcounter = 0
        slim_counter = 0
        factorCounter = 0
        tfidf_counter = 0

        for knn, shrink, recommender in zip(topK, shrink, self.recommender_list):
            if recommender.__class__ in [P3alphaRecommender]:
                if type(similarity_args["alphaP3"]) is not list:
                    similarity_args["alphaP3"] = [similarity_args["alphaP3"]]
                recommender.fit(topK=knn, alpha=similarity_args["alphaP3"][p3counter], min_rating=0, implicit=True,
                                normalize_similarity=True, force_compute_sim=force_compute_sim)
                p3counter += 1

            elif recommender.__class__ in [RP3betaRecommender]:
                recommender.fit(alpha=similarity_args["alphaRP3"][rp3bcounter],
                                beta=similarity_args["betaRP"][rp3bcounter], min_rating=0,
                                topK=knn, implicit=True, normalize_similarity=True, force_compute_sim=force_compute_sim)
                rp3bcounter += 1

            elif recommender.__class__ in [ItemKNNCBFRecommender]:
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim,
                                feature_weighting_index=similarity_args["feature_weighting_index"])

            elif recommender.__class__ in [SLIMElasticNetRecommender]:
                recommender.fit(topK=knn, l1_ratio=similarity_args["l1_ratio"], force_compute_sim=force_compute_sim)

            elif recommender.__class__ in [ItemKNNCFRecommender]:
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim,
                                tfidf=similarity_args["tfidf"][tfidf_counter])
                tfidf_counter += 1

            else:  # ItemCF, UserCF, ItemCBF, UserCBF
                recommender.fit(knn, shrink, force_compute_sim=force_compute_sim)

    def change_weights(self, level, pop):
        if level < pop[0]:
            # return [0, 0, 0, 0, 0, 0, 0, 0]
            return self.d_weights[0]
            # return [0.45590938562950867, 0, 0.23905548168035573, 0.017005850670624212, 0.9443556793576228, 0.19081956929601618, 0, 0.11267140391070507]

        elif pop[0] < level < pop[1]:
            # return self.weights
            # return [0, 0, 0, 0, 0, 0, 0, 0]
            # return [0.973259052781316, 0, 0.8477517414017691, 0.33288193455193427, 0.9696801027638645, 0.4723616073494711, 0, 0.4188403112229081]
            return self.d_weights[1]
        else:
            # return self.weights
            # return [0, 0, 0, 0, 0, 0, 0, 0]
            return self.d_weights[2]
            # return [0.9780713488404191, 0, 0.9694246318172682, 0.5703399158380364, 0.9721597253259535, 0.9504112133900943, 0, 0.9034510004379944]

    def recommend(self, user_id_array, dict_pop=None, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False):

        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        weights = self.weights
        if cutoff == None:
            # noinspection PyUnresolvedReferences
            cutoff = self.URM_train.shape[1] - 1
        else:
            cutoff

        # compute the scores using the dot product
        # noinspection PyUnresolvedReferences
        if self.sparse_weights:
            scores = []
            # noinspection PyUnresolvedReferences
            for recommender in self.recommender_list:
                if recommender.__class__ in [HybridRecommender]:
                    scores.append(self.compute_score_hybrid(recommender, user_id_array, dict_pop,
                                                            remove_seen_flag=True, remove_top_pop_flag=False,
                                                            remove_CustomItems_flag=False))

                    continue
                scores_batch = recommender.compute_item_score(user_id_array)
                # scores_batch = np.ravel(scores_batch) # because i'm not using batch

                for user_index in range(len(user_id_array)):

                    user_id = user_id_array[user_index]

                    if remove_seen_flag:
                        scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

                if remove_top_pop_flag:
                    scores_batch = self._remove_TopPop_on_scores(scores_batch)

                if remove_CustomItems_flag:
                    scores_batch = self._remove_CustomItems_on_scores(scores_batch)

                scores.append(scores_batch)

            final_score = np.zeros(scores[0].shape)

            if self.dynamic:
                for user_index in range(len(user_id_array)):
                    user_id = user_id_array[user_index]
                    user_profile = self.URM_train.indices[
                                   self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                    if self.onPop:
                        level = int(playlist_popularity(user_profile, dict_pop))
                    else:
                        level = int(lenght_playlist(user_profile))
                    # weights = self.change_weights(user_id)
                    weights = self.change_weights(level, self.pop)
                    assert len(weights) == len(scores), "Scores and weights have different lengths"

                    final_score_line = np.zeros(scores[0].shape[1])
                    if sum(weights) > 0:
                        for score, weight in zip(scores, weights):
                            final_score_line += score[user_index] * weight
                    final_score[user_index] = final_score_line
            else:
                for score, weight in zip(scores, weights):
                    final_score += (score * weight)

        else:
            raise NotImplementedError

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-final_score).argpartition(cutoff, axis=1)[:, 0:cutoff]

        relevant_items_partition_original_value = final_score[
            np.arange(final_score.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        # scores = final_score
        # # relevant_items_partition is block_size x cutoff
        # relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]
        #
        # relevant_items_partition_original_value = scores_batch[
        #     np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        # relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        # ranking = relevant_items_partition[
        #     np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()

        # Creating numpy array for training XGBoost



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking


class SLIMElasticNetRecommender(SimilarityMatrixRecommender, Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train):

        super(SLIMElasticNetRecommender, self).__init__()

        self.URM_train = URM_train

    def fit(self, l1_ratio=0.1, positive_only=True, topK=100, force_compute_sim=True):

        assert 0 <= l1_ratio <= 1, "SLIM_ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(
            l1_ratio)

        if not force_compute_sim:
            found = True
            try:
                with open(os.path.join("SLIM_ElasticNet_Matrix.pkl"), 'rb') as handle:
                    (W_sparse_new) = pickle.load(handle)
            except FileNotFoundError:
                found = False

            if found:
                self.W_sparse = W_sparse_new
                print("Saved ElasticNet SLIM Matrix Used!")
                return

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=80,
                                tol=1e-4)

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            if time.time() - start_time_printBatch > 30 or currentItem == n_items - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(
                    currentItem + 1,
                    100.0 * float(currentItem + 1) / n_items,
                    (time.time() - start_time) / 60,
                    float(currentItem) / (time.time() - start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)

        with open(os.path.join("SLIM_ElasticNet_Matrix.pkl"), 'wb') as handle:
            pickle.dump(self.W_sparse, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


start_time = time.time()
if __name__ == '__main__':
    dataReader = RS_Data_Loader(distr_split=False)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()

    evaluator = SequentialEvaluator(URM_test, URM_train, exclude_seen=True)
    evaluator_validation_wrapper = EvaluatorWrapper(evaluator)

    recommender_class = HybridRecommender

    n_cases = 100
    metric_to_optimize = 'MAP'

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
        # PureSVDRecommender,

        SLIMElasticNetRecommender
    ]

    this_output_root_path = "Hybrid_no_intervals:" + "{}".format(
        "_".join([x.RECOMMENDER_NAME for x in recommender_list]))

    # since test and validation are the same for now, here I don't pass the evaluator test (otherwise it also crash)
    parameterSearch = BayesianSearch(recommender_class, evaluator_validation_wrapper)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["weights1"] = range(0, 1)
    hyperparamethers_range_dictionary["weights2"] = range(0, 1)
    hyperparamethers_range_dictionary["weights3"] = range(0, 1)
    hyperparamethers_range_dictionary["weights4"] = range(0, 1)
    hyperparamethers_range_dictionary["weights5"] = range(0, 1)
    hyperparamethers_range_dictionary["weights6"] = range(0, 1)
    hyperparamethers_range_dictionary["weights7"] = range(0, 1)
    # hyperparamethers_range_dictionary["weights8"] = range(0, 1)
    # hyperparamethers_range_dictionary["weights7"] = list(np.linspace(0, 1, 10))  # range(0, 1)
    # hyperparamethers_range_dictionary["weights8"] = list(np.linspace(0, 1, 10))  # range(0, 1)
    # hyperparamethers_range_dictionary["pop1"] = list(range(80, 200))  # list(np.linspace(0, 1, 11))
    # hyperparamethers_range_dictionary["pop2"] = list(range(250, 450))  # list(np.linspace(0, 1, 11))


    lambda_i = 0.03868852215907281
    lambda_j = 0.020779886132281544
    old_similrity_matrix = None
    num_factors = 95
    l1_ratio = 1e-06

    alphaRP3_1 = 0.457685370741483
    betaRP_1 = 0.289432865731463
    lambda_i_1 = 0.6892356201296567
    lambda_j_1 = 0.35586838378889707

    alphaRP3_3 = 0.49774549098196397
    betaRP_3 = 0.2333486973947896
    lambda_i_3 = 0.10467537896611145
    lambda_j_3 = 0.004454204678491891
    num_factors_3 = 95

    dynamic_best = [
        [0.4, 0.03863232277574469, 0.008527738266632112, 0.2560912624445676, 0.7851755932819731, 0.4112843940329439],
        [0.2, 0.012499871230102988, 0.020242981888115352, 0.9969708006657074, 0.9999132876156388, 0.6888103295594851],
        [0.2, 0.10389111810225915, 0.14839466129917822, 0.866992903043857, 0.07010619211847613, 0.5873532658846817]
    ]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM, recommender_list],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"URM_validation": URM_test, "dynamic": False},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: {
                                 "topK": [10, 220, 150, 160, 61, 236, 40],
                                 "shrink": [180, 0, 15, 2, -1, -1, -1],
                                 "pop": [130, 346],
                                 "weights": [1, 1, 1, 1],
                                 "force_compute_sim": False,
                                 "feature_weighting_index": 0,
                                 "old_similarity_matrix": old_similrity_matrix,
                                 "epochs": 50,
                                 "lambda_i": [lambda_i_3],
                                 "lambda_j": [lambda_j_3],
                                 "num_factors": num_factors_3,
                                 'alphaP3': [0.5203791059230995],
                                 'alphaRP3': [0.3855771543086173],
                                 'betaRP': [0.5217815631262526],
                                 'l1_ratio': 2.726530612244898e-05,
                                 "weights_to_dweights": -1,
                                 "tfidf": [True, False]},
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = this_output_root_path

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             metric=metric_to_optimize,  # do not put output path
                                             output_root_path="Hybrid",
                                             init_points=90
                                             )
    print(best_parameters)
