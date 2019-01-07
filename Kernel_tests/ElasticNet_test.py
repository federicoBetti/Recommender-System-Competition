import datetime
import os
import sys
import time
import traceback
from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
from scipy import sparse as sps
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet

# from bayes_opt import BayesianOptimization
from ParameterTuning.BayesianOptimization_master.bayes_opt.bayesian_optimization import BayesianOptimization

'''
Elastic Net
'''


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

        # old working version
        '''
        Per come l'abbiamo noi dovrebbe arrivare sempre user_id_array con un solo elemento e quindi cosi funziona,
        per impplementazioni future rifare funzionare il batch
        Ho dovuto fare cosi un user alla volta per far funzionare l'hybrid!
        '''
        # # Compute the scores using the model-specific function
        # # Vectorize over all users in user_id_array
        # scores_batch = self.compute_item_score(user_id_array)
        #
        # for user_index in range(len(user_id_array)):
        #
        #     assert len(user_id_array) == 1, "La lunghezza del user_id_array è {} ( > 1 ) e la versione batch non è " \
        #                                     "ancora stata implementata".format(len(user_id_array))
        #     user_id = user_id_array[user_index]
        #     scores_batch = np.ravel(scores_batch) # only because len(user_id_array) == 1
        #     if remove_seen_flag:
        #         scores_batch = self._remove_seen_on_scores(user_id, scores_batch)
        #
        #     # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        #     # - Partition the data to extract the set of relevant items
        #     # - Sort only the relevant items
        #     # - Get the original item index
        #     # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
        #     # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
        #     # ranking = relevant_items_partition[relevant_items_partition_sorting]
        #     #
        #     # ranking_list.append(ranking)
        #
        #     if remove_top_pop_flag:
        #         scores_batch = self._remove_TopPop_on_scores(scores_batch)
        #
        #     if remove_CustomItems_flag:
        #         scores_batch = self._remove_CustomItems_on_scores(scores_batch)
        #
        #     # scores_batch = np.arange(0,3260).reshape((1, -1))
        #     # scores_batch = np.repeat(scores_batch, 1000, axis = 0)
        #
        #     # relevant_items_partition is block_size x cutoff
        #     relevant_items_partition = (-scores_batch).argpartition(cutoff)[0:cutoff]
        #
        #     # Get original value and sort it
        #     # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        #     # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        #     relevant_items_partition_original_value = scores_batch[relevant_items_partition]
        #     relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value)
        #     ranking = relevant_items_partition[relevant_items_partition_sorting]
        #     # print("Score batch: {}, Relevenat items parition: {}, Ranking: {}".format(scores_batch, relevant_items_partition, ranking))
        #     ranking_list = ranking.tolist()
        #
        #     # # Return single list for one user, instead of list of lists
        #     # if single_user:
        #     #     ranking_list = ranking_list[0]
        #
        #     return ranking_list

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
                                max_iter=30,
                                tol=1e-4)

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = self.URM_train.shape[1]

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
            y = self.URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = self.URM_train.indptr[currentItem]
            end_pos = self.URM_train.indptr[currentItem + 1]

            current_item_data_backup = self.URM_train.data[start_pos: end_pos].copy()
            self.URM_train.data[start_pos: end_pos] = 0.0

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
            self.URM_train.data[start_pos:end_pos] = current_item_data_backup

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
        self.W_sparse = csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                   shape=(n_items, n_items), dtype=np.float32)


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

        best_solution = self.bayesian_optimizer.res['max']

        self.best_solution_val = best_solution["max_val"]
        self.best_solution_parameters = best_solution["max_params"].copy()
        self.best_solution_parameters = self.parameter_bayesian_to_token(self.best_solution_parameters)
        self.best_solution_parameters = self.from_fit_params_to_saved_params[
            frozenset(self.best_solution_parameters.items())]

        print("BayesianSearch: Best config is: Config {}, {} value is {:.4f}\n".format(
            self.best_solution_parameters, metric, self.best_solution_val))

        return self.best_solution_parameters.copy()

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
        # self.train = pd.read_csv(os.path.join("../input", "train.csv"))
        # self.tracks = pd.read_csv(os.path.join("../input", "tracks.csv"))
        # self.target_playlist = pd.read_csv(os.path.join("../input", "target_playlists.csv"))
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


start_time = time.time()
if __name__ == '__main__':
    evaluate_algorithm = True
    delete_old_computations = False
    slim_after_hybrid = False

    # delete_previous_intermediate_computations()
    # if not evaluate_algorithm:
    #     delete_previous_intermediate_computations()
    # else:
    #     print("ATTENTION: old intermediate computations kept, pay attention if running with all_train")

    dataReader = RS_Data_Loader(all_train=not evaluate_algorithm, distr_split=False)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()

    evaluator = SequentialEvaluator(URM_test, URM_train, exclude_seen=True)
    evaluator_validation_wrapper = EvaluatorWrapper(evaluator)

    recommender_class = SLIMElasticNetRecommender

    # On pop it used to choose if have dynamic weights for
    recommender = recommender_class(URM_train)

    n_cases = 30
    metric_to_optimize = 'MAP'

    parameterSearch = BayesianSearch(recommender_class, evaluator_validation_wrapper)
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = list(range(10, 500, 30))
    hyperparamethers_range_dictionary["l1_ratio"] = list(np.linspace(0.000001, 0.0001, 50))

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             metric=metric_to_optimize,  # do not put output path
                                             output_root_path="ElasticNet",
                                             init_points=15
                                             )
    print(best_parameters)
