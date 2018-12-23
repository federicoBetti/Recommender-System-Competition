import os
from enum import Enum

import scipy.sparse as sps
import sys
from sklearn.preprocessing import normalize
import numpy as np
import pickle, time
import pandas as pd
from scipy.sparse import csr_matrix
import scipy
from scipy import sparse
import random

import datetime

from functools import partial
import traceback, pickle
import numpy as np
import copy


class BayesianOptimization(object):

    def __init__(self, f, pbounds, random_state=None, verbose=1):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        """
        # Store the original dictionary
        self.pbounds = pbounds

        self.random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self.space = TargetSpace(f, pbounds, random_state)

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Counter of iterations
        self.i = 0

        # Internal GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=self.random_state
        )

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.space.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}

        # non-public config for maximizing the aquisition function
        # (used to speedup tests, but generally leave these as is)
        self._acqkw = {'n_warmup': 100000, 'n_iter': 250}

        # Verbose
        self.verbose = verbose

    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """
        # Concatenate new random points to possible existing
        # points from self.explore method.

        rand_points = self.space.random_points(init_points)
        self.init_points.extend(rand_points)

        # Evaluate target function at all initialization points
        count = 1
        for x in self.init_points:
            print("Initial test number {}/{} to understand best starting points...".format(count, len(self.init_points)))
            y = self._observe_point(x)
            count += 1
            gc.collect()

        # Add the points from `self.initialize` to the observations
        if self.x_init:
            x_init = np.vstack(self.x_init)
            y_init = np.hstack(self.y_init)
            for x, y in zip(x_init, y_init):
                self.space.add_observation(x, y)
                if self.verbose:
                    self.plog.print_step(x, y)

        # Updates the flag
        self.initialized = True

    def _observe_point(self, x):
        y = self.space.observe_point(x)
        if self.verbose:
            self.plog.print_step(x, y)
        return y

    def explore(self, points_dict, eager=False):
        """Method to explore user defined points.

        :param points_dict:
        :param eager: if True, these points are evaulated immediately
        """
        if eager:
            self.plog.reset_timer()
            if self.verbose:
                self.plog.print_header(initialization=True)

            points = self.space._dict_to_points(points_dict)
            for x in points:
                self._observe_point(x)
        else:
            points = self.space._dict_to_points(points_dict)
            self.init_points = points

    def initialize(self, points_dict):
        """
        Method to introduce points for which the target function value is known

        :param points_dict:
            dictionary with self.keys and 'target' as keys, and list of
            corresponding values as values.

        ex:
            {
                'target': [-1166.19102, -1142.71370, -1138.68293],
                'alpha': [7.0034, 6.6186, 6.0798],
                'colsample_bytree': [0.6849, 0.7314, 0.9540],
                'gamma': [8.3673, 3.5455, 2.3281],
            }

        :return:
        """

        self.y_init.extend(points_dict['target'])
        for i in range(len(points_dict['target'])):
            all_points = []
            for key in self.space.keys:
                all_points.append(points_dict[key][i])
            self.x_init.append(all_points)

    def initialize_df(self, points_df):
        """
        Method to introduce point for which the target function
        value is known from pandas dataframe file

        :param points_df:
            pandas dataframe with columns (target, {list of columns matching
            self.keys})

        ex:
              target        alpha      colsample_bytree        gamma
        -1166.19102       7.0034                0.6849       8.3673
        -1142.71370       6.6186                0.7314       3.5455
        -1138.68293       6.0798                0.9540       2.3281
        -1146.65974       2.4566                0.9290       0.3456
        -1160.32854       1.9821                0.5298       8.7863

        :return:
        """

        for i in points_df.index:
            self.y_init.append(points_df.loc[i, 'target'])

            all_points = []
            for key in self.space.keys:
                all_points.append(points_df.loc[i, key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """
        # Update the internal object stored dict
        self.pbounds.update(new_bounds)
        self.space.set_bounds(new_bounds)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        :param acq:
            Acquisition function to be used, defaults to Upper Confidence Bound.

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        Returns
        -------
        :return: Nothing

        Example:
        >>> xs = np.linspace(-2, 10, 10000)
        >>> f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2/10) + 1/ (xs**2 + 1)
        >>> bo = BayesianOptimization(f=lambda x: f[int(x)],
        >>>                           pbounds={"x": (0, len(f)-1)})
        >>> bo.maximize(init_points=2, n_iter=25, acq="ucb", kappa=1)
        """

        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        y_max = self.space.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        self.gp.fit(self.space.X, self.space.Y)

        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.space.bounds,
                        random_state=self.random_state,
                        **self._acqkw)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            print("New iterations for parameter testing, number {}/{}".format(i, n_iter))
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            while x_max in self.space:
                x_max = self.space.random_points(1)[0]
                pwarning = True

            # Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            if self.verbose:
                self.plog.print_step(x_max, y, pwarning)

            # Updating the GP.
            self.gp.fit(self.space.X, self.space.Y)

            # Update the best params seen so far
            self.res['max'] = self.space.max_point()
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys, x_max)))

            # Update maximum value to search for next probe point.
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.space.bounds,
                            random_state=self.random_state,
                            **self._acqkw)

            # Keep track of total number of iterations
            self.i = i
            gc.collect()

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

    def points_to_csv(self, file_name):
        """
        After training all points for which we know target variable
        (both from initialization and optimization) are saved

        :param file_name: name of the file where points will be saved in the csv
            format

        :return: None
        """

        points = np.hstack((self.space.X, np.expand_dims(self.space.Y, axis=1)))
        header = ','.join(self.space.keys + ['target'])
        np.savetxt(file_name, points, header=header, delimiter=',', comments='')

    # --- API compatibility ---

    @property
    def X(self):
        return self.space.X

    @property
    def Y(self):
        return self.space.Y

    @property
    def keys(self):
        return self.space.keys

    @property
    def f(self):
        return self.space.target_func

    @property
    def bounds(self):
        return self.space.bounds

    @property
    def dim(self):
        return self.space.dim


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


def pp_metrics(metric_names, metric_values, metric_at):
    """
    Pretty-prints metric values
    :param metrics_arr:
    :return:
    """
    assert len(metric_names) == len(metric_values)
    if isinstance(metric_at, int):
        metric_at = [metric_at] * len(metric_values)
    return ' '.join(['{}: {:.4f}'.format(mname, mvalue) if mcutoff is None or mcutoff == 0 else
                     '{}@{}: {:.4f}'.format(mname, mcutoff, mvalue)
                     for mname, mcutoff, mvalue in zip(metric_names, metric_at, metric_values)])


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


class Metrics_Object(object):
    """
    Abstract class that should be used as superclass of all metrics requiring an object, therefore a state, to be computed
    """

    def __init__(self):
        pass

    def add_recommendations(self, recommended_items_ids):
        raise NotImplementedError()

    def get_metric_value(self):
        raise NotImplementedError()

    def merge_with_other(self, other_metric_object):
        raise NotImplementedError()


class Coverage_Item(Metrics_Object):
    """
    Item coverage represents the percentage of the overall items which were recommended
    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_items, ignore_items):
        super(Coverage_Item, self).__init__()
        self.recommended_mask = np.zeros(n_items, dtype=np.bool)
        self.n_ignore_items = len(ignore_items)

    def add_recommendations(self, recommended_items_ids):
        self.recommended_mask[recommended_items_ids] = True

    def get_metric_value(self):
        return self.recommended_mask.sum() / (len(self.recommended_mask) - self.n_ignore_items)

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Coverage_Item, "Coverage_Item: attempting to merge with a metric object of different type"

        self.recommended_mask = np.logical_or(self.recommended_mask, other_metric_object.recommended_mask)


class Coverage_User(Metrics_Object):
    """
    User coverage represents the percentage of the overall users for which we can make recommendations.
    If there is at least one recommendation the user is considered as covered
    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_users, ignore_users):
        super(Coverage_User, self).__init__()
        self.users_mask = np.zeros(n_users, dtype=np.bool)
        self.n_ignore_users = len(ignore_users)

    def add_recommendations(self, recommended_items_ids, user_id):
        self.users_mask[user_id] = len(recommended_items_ids) > 0

    def get_metric_value(self):
        return self.users_mask.sum() / (len(self.users_mask) - self.n_ignore_users)

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Coverage_User, "Coverage_User: attempting to merge with a metric object of different type"

        self.users_mask = np.logical_or(self.users_mask, other_metric_object.users_mask)


class Gini_Diversity(Metrics_Object):
    """
    Gini diversity index, computed from the Gini Index but with inverted range, such that high values mean higher diversity
    This implementation ignores zero-occurrence items

    # From https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    #
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8174&rep=rep1&type=pdf
    """

    def __init__(self, n_items, ignore_items):
        super(Gini_Diversity, self).__init__()
        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        self.recommended_counter[recommended_items_ids] += 1

    def get_metric_value(self):
        recommended_counter = self.recommended_counter.copy()

        recommended_counter_mask = np.ones_like(recommended_counter, dtype=np.bool)
        recommended_counter_mask[self.ignore_items] = False
        recommended_counter_mask[recommended_counter == 0] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        n_items = len(recommended_counter)

        recommended_counter_sorted = np.sort(recommended_counter)  # values must be sorted
        index = np.arange(1, n_items + 1)  # index per array element

        # gini_index = (np.sum((2 * index - n_items  - 1) * recommended_counter_sorted)) / (n_items * np.sum(recommended_counter_sorted))
        gini_diversity = 2 * np.sum(
            (n_items + 1 - index) / (n_items + 1) * recommended_counter_sorted / np.sum(recommended_counter_sorted))

        return gini_diversity

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Gini_Diversity, "Gini_Diversity: attempting to merge with a metric object of different type"

        self.recommended_counter += other_metric_object.recommended_counter


class Diversity_Herfindahl(Metrics_Object):
    """
    The Herfindahl index is also known as Concentration index, it is used in economy to determine whether the market quotas
    are such that an excessive concentration exists. It is here used as a diversity index, if high means high diversity.

    It is known to have a small value range in recommender systems, between 0.9 and 1.0

    The Herfindahl index is a function of the square of the probability an item has been recommended to any user, hence
    The Herfindahl index is equivalent to MeanInterList diversity as they measure the same quantity.

    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8174&rep=rep1&type=pdf
    """

    def __init__(self, n_items, ignore_items):
        super(Diversity_Herfindahl, self).__init__()
        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        self.recommended_counter[recommended_items_ids] += 1

    def get_metric_value(self):
        recommended_counter = self.recommended_counter.copy()

        recommended_counter_mask = np.ones_like(recommended_counter, dtype=np.bool)
        recommended_counter_mask[self.ignore_items] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        herfindahl_index = 1 - np.sum((recommended_counter / recommended_counter.sum()) ** 2)

        return herfindahl_index

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Diversity_Herfindahl, "Diversity_Herfindahl: attempting to merge with a metric object of different type"

        self.recommended_counter += other_metric_object.recommended_counter


class Shannon_Entropy(Metrics_Object):
    """
    Shannon Entropy is a well known metric to measure the amount of information of a certain string of data.
    Here is applied to the global number of times an item has been recommended.

    It has a lower bound and can reach values over 12.0 for random recommenders.
    A high entropy means that the distribution is random uniform across all users.

    Note that while a random uniform distribution
    (hence all items with SIMILAR number of occurrences)
    will be highly diverse and have high entropy, a perfectly uniform distribution
    (hence all items with EXACTLY IDENTICAL number of occurrences)
    will have 0.0 entropy while being the most diverse possible.

    """

    def __init__(self, n_items, ignore_items):
        super(Shannon_Entropy, self).__init__()
        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        self.recommended_counter[recommended_items_ids] += 1

    def get_metric_value(self):
        assert np.all(
            self.recommended_counter >= 0.0), "Shannon_Entropy: self.recommended_counter contains negative counts"

        recommended_counter = self.recommended_counter.copy()

        # Ignore from the computation both ignored items and items with zero occurrence.
        # Zero occurrence items will have zero probability and will not change the result, butt will generate nans if used in the log
        recommended_counter_mask = np.ones_like(recommended_counter, dtype=np.bool)
        recommended_counter_mask[self.ignore_items] = False
        recommended_counter_mask[recommended_counter == 0] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        n_recommendations = recommended_counter.sum()

        recommended_probability = recommended_counter / n_recommendations

        shannon_entropy = -np.sum(recommended_probability * np.log2(recommended_probability))

        return shannon_entropy

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Gini_Diversity, "Shannon_Entropy: attempting to merge with a metric object of different type"

        assert np.all(
            self.recommended_counter >= 0.0), "Shannon_Entropy: self.recommended_counter contains negative counts"
        assert np.all(
            other_metric_object.recommended_counter >= 0.0), "Shannon_Entropy: other.recommended_counter contains negative counts"

        self.recommended_counter += other_metric_object.recommended_counter


class Novelty(Metrics_Object):
    """
    Novelty measures how "novel" a recommendation is in terms of how popular the item was in the train set.

    Due to this definition, the novelty of a cold item (i.e. with no interactions in the train set) is not defined,
    in this implementation cold items are ignored and their contribution to the novelty is 0.

    A recommender with high novelty will be able to recommend also long queue (i.e. unpopular) items.

    Mean self-information  (Zhou 2010)
    """

    def __init__(self, URM_train):
        super(Novelty, self).__init__()

        URM_train = sps.csc_matrix(URM_train)
        URM_train.eliminate_zeros()
        self.item_popularity = np.ediff1d(URM_train.indptr)

        self.novelty = 0.0
        self.n_evaluated_users = 0
        self.n_items = len(self.item_popularity)
        self.n_interactions = self.item_popularity.sum()

    def add_recommendations(self, recommended_items_ids):
        recommended_items_popularity = self.item_popularity[recommended_items_ids]

        probability = recommended_items_popularity / self.n_interactions
        probability = probability[probability != 0]

        self.novelty += np.sum(-np.log2(probability) / self.n_items)
        self.n_evaluated_users += 1

    def get_metric_value(self):
        if self.n_evaluated_users == 0:
            return 0.0

        return self.novelty / self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Novelty, "Novelty: attempting to merge with a metric object of different type"

        self.novelty = self.novelty + other_metric_object.novelty
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users


class Diversity_similarity(Metrics_Object):
    """
    Intra list diversity computes the diversity of items appearing in the recommendations received by each single user, by using an item_diversity_matrix.

    It can be used, for example, to compute the diversity in terms of features for a collaborative recommender.

    A content-based recommender will have low IntraList diversity if that is computed on the same features the recommender uses.
    A TopPopular recommender may exhibit high IntraList diversity.

    """

    def __init__(self, item_diversity_matrix):
        super(Diversity_similarity, self).__init__()

        assert np.all(item_diversity_matrix >= 0.0) and np.all(item_diversity_matrix <= 1.0), \
            "item_diversity_matrix contains value greated than 1.0 or lower than 0.0"

        self.item_diversity_matrix = item_diversity_matrix

        self.n_evaluated_users = 0
        self.diversity = 0.0

    def add_recommendations(self, recommended_items_ids):

        current_recommended_items_diversity = 0.0

        for item_index in range(len(recommended_items_ids) - 1):
            item_id = recommended_items_ids[item_index]

            item_other_diversity = self.item_diversity_matrix[item_id, recommended_items_ids]
            item_other_diversity[item_index] = 0.0

            current_recommended_items_diversity += np.sum(item_other_diversity)

        self.diversity += current_recommended_items_diversity / (
            len(recommended_items_ids) * (len(recommended_items_ids) - 1))

        self.n_evaluated_users += 1

    def get_metric_value(self):

        if self.n_evaluated_users == 0:
            return 0.0

        return self.diversity / self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Diversity_similarity, "Diversity: attempting to merge with a metric object of different type"

        self.diversity = self.diversity + other_metric_object.diversity
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users


class Diversity_MeanInterList(Metrics_Object):
    """
    MeanInterList diversity measures the uniqueness of different users' recommendation lists.

    It can be used to measure how "diversified" are the recommendations different users receive.

    While the original proposal called this metric "Personalization", we do not use this name since the highest MeanInterList diversity
    is exhibited by a non personalized Random recommender.

    It can be demonstrated that this metric does not require to compute the common items all possible couples of users have in common
    but rather it is only sensitive to the total amount of time each item has been recommended.

    MeanInterList diversity is a function of the square of the probability an item has been recommended to any user, hence
    MeanInterList diversity is equivalent to the Herfindahl index as they measure the same quantity.

    A TopPopular recommender that does not remove seen items will have 0.0 MeanInterList diversity.


    pag. 3, http://www.pnas.org/content/pnas/107/10/4511.full.pdf

    @article{zhou2010solving,
      title={Solving the apparent diversity-accuracy dilemma of recommender systems},
      author={Zhou, Tao and Kuscsik, Zolt{\'a}n and Liu, Jian-Guo and Medo, Mat{\'u}{\v{s}} and Wakeling, Joseph Rushton and Zhang, Yi-Cheng},
      journal={Proceedings of the National Academy of Sciences},
      volume={107},
      number={10},
      pages={4511--4515},
      year={2010},
      publisher={National Acad Sciences}
    }

    # The formula is diversity_cumulative += 1 - common_recommendations(user1, user2)/cutoff
    # for each couple of users, except the diagonal. It is VERY computationally expensive
    # We can move the 1 and cutoff outside of the summation. Remember to exclude the diagonal
    # co_counts = URM_predicted.dot(URM_predicted.T)
    # co_counts[np.arange(0, n_user, dtype=np.int):np.arange(0, n_user, dtype=np.int)] = 0
    # diversity = (n_user**2 - n_user) - co_counts.sum()/self.cutoff

    # If we represent the summation of co_counts separating it for each item, we will have:
    # co_counts.sum() = co_counts_item1.sum()  + co_counts_item2.sum() ...
    # If we know how many times an item has been recommended, co_counts_item1.sum() can be computed as how many couples of
    # users have item1 in common. If item1 has been recommended n times, the number of couples is n*(n-1)
    # Therefore we can compute co_counts.sum() value as:
    # np.sum(np.multiply(item-occurrence, item-occurrence-1))

    # The naive implementation URM_predicted.dot(URM_predicted.T) might require an hour of computation
    # The last implementation has a negligible computational time even for very big datasets

    """

    def __init__(self, n_items, cutoff):
        super(Diversity_MeanInterList, self).__init__()

        self.recommended_counter = np.zeros(n_items, dtype=np.float)

        self.n_evaluated_users = 0
        self.n_items = n_items
        self.diversity = 0.0
        self.cutoff = cutoff

    def add_recommendations(self, recommended_items_ids):
        assert len(
            recommended_items_ids) <= self.cutoff, "Diversity_MeanInterList: recommended list is contains more elements than cutoff"

        self.recommended_counter[recommended_items_ids] += 1
        self.n_evaluated_users += 1

    def get_metric_value(self):
        # Requires to compute the number of common elements for all couples of users
        if self.n_evaluated_users == 0:
            return 1.0

        cooccurrences_cumulative = np.sum(self.recommended_counter ** 2) - self.n_evaluated_users * self.cutoff

        # All user combinations except diagonal
        all_user_couples_count = self.n_evaluated_users ** 2 - self.n_evaluated_users

        diversity_cumulative = all_user_couples_count - cooccurrences_cumulative / self.cutoff

        self.diversity = diversity_cumulative / all_user_couples_count

        return self.diversity

    def get_theoretical_max(self):
        global_co_occurrence_count = (
                                         self.n_evaluated_users * self.cutoff) ** 2 / self.n_items - self.n_evaluated_users * self.cutoff

        mild = 1 - 1 / (self.n_evaluated_users ** 2 - self.n_evaluated_users) * (
            global_co_occurrence_count / self.cutoff)

        return mild

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Diversity_MeanInterList, "Diversity_MeanInterList: attempting to merge with a metric object of different type"

        assert np.all(
            self.recommended_counter >= 0.0), "Diversity_MeanInterList: self.recommended_counter contains negative counts"
        assert np.all(
            other_metric_object.recommended_counter >= 0.0), "Diversity_MeanInterList: other.recommended_counter contains negative counts"

        self.recommended_counter += other_metric_object.recommended_counter
        self.n_evaluated_users += other_metric_object.n_evaluated_users


def create_empty_metrics_dict(n_items, n_users, URM_train, ignore_items, ignore_users, cutoff,
                              diversity_similarity_object):
    empty_dict = {}

    # from Base.Evaluation.ResultMetric import ResultMetric
    # empty_dict = ResultMetric()

    for metric in EvaluatorMetrics:
        if metric == EvaluatorMetrics.COVERAGE_ITEM:
            empty_dict[metric.value] = Coverage_Item(n_items, ignore_items)

        elif metric == EvaluatorMetrics.DIVERSITY_GINI:
            empty_dict[metric.value] = Gini_Diversity(n_items, ignore_items)

        elif metric == EvaluatorMetrics.SHANNON_ENTROPY:
            empty_dict[metric.value] = Shannon_Entropy(n_items, ignore_items)

        elif metric == EvaluatorMetrics.COVERAGE_USER:
            empty_dict[metric.value] = Coverage_User(n_users, ignore_users)

        elif metric == EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST:
            empty_dict[metric.value] = Diversity_MeanInterList(n_items, cutoff)

        elif metric == EvaluatorMetrics.DIVERSITY_HERFINDAHL:
            empty_dict[metric.value] = Diversity_Herfindahl(n_items, ignore_items)

        elif metric == EvaluatorMetrics.NOVELTY:
            empty_dict[metric.value] = Novelty(URM_train)

        elif metric == EvaluatorMetrics.DIVERSITY_SIMILARITY:
            if diversity_similarity_object is not None:
                empty_dict[metric.value] = copy.deepcopy(diversity_similarity_object)
        else:
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
            URM_test = sps.csr_matrix(URM_test)
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

    def _run_evaluation_on_selected_users(self, recommender_object, usersToEvaluate):

        start_time = time.time()
        start_time_print = time.time()

        results_dict = {}

        for cutoff in self.cutoff_list:
            results_dict[cutoff] = create_empty_metrics_dict(self.n_items, self.n_users,
                                                             recommender_object.URM_train,
                                                             self.ignore_items_ID,
                                                             self.ignore_users_ID,
                                                             cutoff,
                                                             self.diversity_object)

        n_users_evaluated = 0

        for test_user in usersToEvaluate:

            # Being the URM CSR, the indices are the non-zero column indexes
            relevant_items = self.get_user_relevant_items(test_user)

            n_users_evaluated += 1

            recommended_items = recommender_object.recommend(test_user, remove_seen_flag=self.exclude_seen,
                                                             cutoff=self.max_cutoff, remove_top_pop_flag=False,
                                                             remove_CustomItems_flag=self.ignore_items_flag)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                is_relevant_current_cutoff = is_relevant[0:cutoff]
                recommended_items_current_cutoff = recommended_items[0:cutoff]

                results_current_cutoff[EvaluatorMetrics.ROC_AUC.value] += roc_auc(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.PRECISION.value] += precision(is_relevant_current_cutoff,
                                                                                      len(relevant_items))
                results_current_cutoff[EvaluatorMetrics.RECALL.value] += recall(is_relevant_current_cutoff,
                                                                                relevant_items)
                results_current_cutoff[EvaluatorMetrics.RECALL_TEST_LEN.value] += recall_min_test_len(
                    is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.MAP.value] += map(is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.MRR.value] += rr(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.NDCG.value] += ndcg(recommended_items_current_cutoff,
                                                                            relevant_items,
                                                                            relevance=self.get_user_test_ratings(
                                                                                test_user), at=cutoff)
                results_current_cutoff[EvaluatorMetrics.HIT_RATE.value] += is_relevant_current_cutoff.sum()
                results_current_cutoff[EvaluatorMetrics.ARHR.value] += arhr(is_relevant_current_cutoff)

                results_current_cutoff[EvaluatorMetrics.NOVELTY.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_GINI.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.SHANNON_ENTROPY.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_USER.value].add_recommendations(
                    recommended_items_current_cutoff, test_user)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].add_recommendations(
                    recommended_items_current_cutoff)

                if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in results_current_cutoff:
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_SIMILARITY.value].add_recommendations(
                        recommended_items_current_cutoff)

            if time.time() - start_time_print > 30 or n_users_evaluated == len(self.usersToEvaluate):
                print(
                    "SequentialEvaluator: Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                        n_users_evaluated,
                        100.0 * float(n_users_evaluated) / len(self.usersToEvaluate),
                        time.time() - start_time,
                        float(n_users_evaluated) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print = time.time()

        return results_dict, n_users_evaluated


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
            results_dict[cutoff] = create_empty_metrics_dict(self.n_items, self.n_users,
                                                             recommender_object.get_URM_train(),
                                                             self.ignore_items_ID,
                                                             self.ignore_users_ID,
                                                             cutoff,
                                                             self.diversity_object)

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
                                                                        remove_CustomItems_flag=self.ignore_items_flag)

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

                    results_current_cutoff[EvaluatorMetrics.NOVELTY.value].add_recommendations(
                        recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_GINI.value].add_recommendations(
                        recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.SHANNON_ENTROPY.value].add_recommendations(
                        recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM.value].add_recommendations(
                        recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.COVERAGE_USER.value].add_recommendations(
                        recommended_items_current_cutoff, user_id)
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].add_recommendations(
                        recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].add_recommendations(
                        recommended_items_current_cutoff)

                    if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in results_current_cutoff:
                        results_current_cutoff[EvaluatorMetrics.DIVERSITY_SIMILARITY.value].add_recommendations(
                            recommended_items_current_cutoff)

                # create both data structures for plotting: lenght and popularity
                if time.time() - start_time_print > 30 or n_users_evaluated == len(self.usersToEvaluate):
                    print(
                        "SequentialEvaluator: Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                            n_users_evaluated,
                            100.0 * float(n_users_evaluated) / len(self.usersToEvaluate),
                            time.time() - start_time,
                            float(n_users_evaluated) / (time.time() - start_time)))

                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_print = time.time()

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

                    if isinstance(value, Metrics_Object):
                        results_current_cutoff[key] = value.get_metric_value()
                    else:
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


import multiprocessing
from functools import partial


class _ParallelEvaluator_batch(Evaluator):
    """SequentialEvaluator"""

    EVALUATOR_NAME = "SequentialEvaluator_Class"

    def __init__(self, URM_test_list, cutoff_list, minRatingsPerUser=1, exclude_seen=True,
                 diversity_object=None,
                 ignore_items=None,
                 ignore_users=None):
        super(_ParallelEvaluator_batch, self).__init__(URM_test_list, cutoff_list,
                                                       diversity_object=diversity_object,
                                                       minRatingsPerUser=minRatingsPerUser, exclude_seen=exclude_seen,
                                                       ignore_items=ignore_items, ignore_users=ignore_users)

    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """
        results_dict, n_users_evaluated = self._run_evaluation_on_selected_users(recommender_object,
                                                                                 self.usersToEvaluate)

        return (results_dict, n_users_evaluated)


def _run_parallel_evaluator(evaluator_object, recommender_object):
    results_dict, _ = evaluator_object.evaluateRecommender(recommender_object)

    return results_dict


def _merge_results_dict(results_dict_1, results_dict_2, n_users_2):
    assert results_dict_1.keys() == results_dict_2.keys(), "_merge_results_dict: the two result dictionaries have different cutoff values"

    merged_dict = copy.deepcopy(results_dict_1)

    for cutoff in merged_dict.keys():

        merged_dict_cutoff = merged_dict[cutoff]
        results_dict_2_cutoff = results_dict_2[cutoff]

        for key in merged_dict_cutoff.keys():

            result_metric = merged_dict_cutoff[key]

            if result_metric is Metrics_Object:
                merged_dict_cutoff[key].merge_with_other(results_dict_2_cutoff[key])
            else:
                merged_dict_cutoff[key] = result_metric + results_dict_2_cutoff[key] * n_users_2


class EvaluatorWrapper(object):
    def __init__(self, evaluator_object):
        self.evaluator_object = evaluator_object

    def evaluateRecommender(self, recommender_object, paramether_dictionary=None):
        return self.evaluator_object.evaluateRecommender(recommender_object)


class DictionaryKeys(Enum):
    # Fields to be filled by caller
    # Dictionary of paramethers needed by the constructor
    CONSTRUCTOR_POSITIONAL_ARGS = 'constructor_positional_args'
    CONSTRUCTOR_KEYWORD_ARGS = 'constructor_keyword_args'

    # List containing all positional arguments needed by the fit function
    FIT_POSITIONAL_ARGS = 'fit_positional_args'
    FIT_KEYWORD_ARGS = 'fit_keyword_args'

    # Contains the dictionary of all keyword args to use for validation
    # With the respectives range
    FIT_RANGE_KEYWORD_ARGS = 'fit_range_keyword_args'

    # Label to be written on log
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


def writeLog(string, logFile):
    print(string)

    if logFile != None:
        logFile.write(string)
        logFile.flush()


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

    def runSingleCase(self, paramether_dictionary_to_evaluate, metric):

        try:

            # Create an object of the same class of the imput
            # Passing the paramether as a dictionary
            recommender = self.recommender_class(*self.dictionary_input[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                                 **self.dictionary_input[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])

            print(self.ALGORITHM_NAME + ": Testing config: {}".format(paramether_dictionary_to_evaluate))

            recommender.fit(*self.dictionary_input[DictionaryKeys.FIT_POSITIONAL_ARGS],
                            **self.dictionary_input[DictionaryKeys.FIT_KEYWORD_ARGS],
                            **paramether_dictionary_to_evaluate)

            # result_dict = self.evaluator_validation(recommender, self.URM_validation, paramether_dictionary_to_evaluate)

            result_dict, _ = self.evaluator_validation.evaluateRecommender(self, paramether_dictionary_to_evaluate)
            result_dict = result_dict[list(result_dict.keys())[0]]

            paramether_dictionary_to_save = self.from_fit_params_to_saved_params_function(recommender,
                                                                                          paramether_dictionary_to_evaluate)
            print("Parameter to save: {}".format(paramether_dictionary_to_save))

            self.from_fit_params_to_saved_params[
                frozenset(paramether_dictionary_to_evaluate.items())] = paramether_dictionary_to_save

            if self.best_solution_val == None or self.best_solution_val < result_dict[metric]:

                writeLog(self.ALGORITHM_NAME + ": New best config found. Config: {} - MAP results: {}\n".format(
                    paramether_dictionary_to_save, result_dict[metric]), self.logFile)

                pickle.dump(paramether_dictionary_to_save.copy(),
                            open(self.output_root_path + "_best_parameters", "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)

                self.best_solution_val = result_dict[metric]
                self.best_solution_parameters = paramether_dictionary_to_save.copy()
                # self.best_solution_object = recommender

                if self.save_best_model:
                    print(self.ALGORITHM_NAME + ": Saving model in {}\n".format(self.output_root_path))
                    recommender.saveModel(self.output_root_path,
                                          file_name=self.recommender_class.RECOMMENDER_NAME + "_best_model")

                if self.evaluator_test is not None:
                    self.evaluate_on_test()

            else:
                writeLog(self.ALGORITHM_NAME + ": Config is suboptimal. Config: {} - MAP results: {}\n".format(
                    paramether_dictionary_to_save, result_dict[metric]), self.logFile)

            return result_dict[metric]


        except Exception as e:

            writeLog(
                self.ALGORITHM_NAME + ": Testing config: {} - Exception {}\n".format(paramether_dictionary_to_evaluate,
                                                                                     str(e)), self.logFile)
            traceback.print_exc()

            return - np.inf

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

        pickle.dump(result_dict.copy(),
                    open(self.output_root_path + "_best_result_test", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        writeLog(self.ALGORITHM_NAME + ": Best result evaluated on URM_test. Config: {} - results: {}\n".format(
            self.best_solution_parameters, result_string), self.logFile)

        return result_dict



def writeLog(string, logFile):
    print(string)

    if logFile != None:
        logFile.write(string)
        logFile.flush()


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
        self.logFile = open(self.output_root_path + "_BayesianSearch.txt", "a")
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

        writeLog("BayesianSearch: Best config is: Config {}, {} value is {:.4f}\n".format(
            self.best_solution_parameters, metric, self.best_solution_val), self.logFile)

        #
        #
        # if folderPath != None:
        #
        #     writeLog("BayesianSearch: Saving model in {}\n".format(folderPath), self.logFile)
        #     self.runSingleCase_param_parsed(dictionary, metric, self.best_solution_parameters, folderPath = folderPath, namePrefix = namePrefix)


        return self.best_solution_parameters.copy()

    #
    # def evaluate_on_test(self):
    #
    #     # Create an object of the same class of the imput
    #     # Passing the paramether as a dictionary
    #     recommender = self.recommender_class(*self.dictionary_input[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
    #                                          **self.dictionary_input[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])
    #
    #
    #     recommender.fit(*self.dictionary_input[DictionaryKeys.FIT_POSITIONAL_ARGS],
    #                     **self.dictionary_input[DictionaryKeys.FIT_KEYWORD_ARGS],
    #                     **self.best_solution_parameters)
    #
    #
    #     result_dict = self.evaluator_test.evaluateRecommender(recommender, self.best_solution_parameters)
    #
    #
    #     writeLog("ParameterSearch: Best result evaluated on URM_test. Config: {} - results: {}\n".format(self.best_solution_parameters, result_dict), self.logFile)
    #
    #     return result_dict
    #



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

            # Always save best model separately
            if self.save_model == "all":
                print(self.ALGORITHM_NAME + ": Saving model in {}\n".format(self.output_root_path))
                recommender.saveModel(self.output_root_path, file_name="_model_{}".format(self.model_counter))

                pickle.dump(paramether_dictionary_to_save.copy(),
                            open(self.output_root_path + "_parameters_{}".format(self.model_counter), "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)

            if self.best_solution_val is None or self.best_solution_val < result_dict[metric]:

                writeLog("BayesianSearch: New best config found. Config: {} - MAP results: {} - time: {}\n".format(
                    paramether_dictionary_to_save, result_dict[metric], datetime.datetime.now()), self.logFile)

                '''REMOVED IN ORDER NOT TO SAVE MODEL PARAMETERS (problem with hybrid and for us useless because
                we'll train with all the train at the end, just memory usage) '''
                # pickle.dump(paramether_dictionary_to_save.copy(),
                #             open(self.output_root_path + "_best_parameters", "wb"),
                #             protocol=pickle.HIGHEST_PROTOCOL)
                #
                # pickle.dump(result_dict.copy(),
                #             open(self.output_root_path + "_best_result_validation", "wb"),
                #             protocol=pickle.HIGHEST_PROTOCOL)
                #
                self.best_solution_val = result_dict[metric]
                # self.best_solution_parameters = paramether_dictionary_to_save.copy()
                # # self.best_solution_object = recommender
                #
                # if self.save_model != "no":
                #     print("BayesianSearch: Saving model in {}\n".format(self.output_root_path))
                #     recommender.saveModel(self.output_root_path, file_name="_best_model")

                if self.evaluator_test is not None:
                    self.evaluate_on_test()

            else:
                writeLog("BayesianSearch: Config is suboptimal. Config: {} - MAP results: {} - time: {}\n".format(
                    paramether_dictionary_to_save, result_dict[metric], datetime.datetime.now()), self.logFile)
            del recommender
            return result_dict[metric]


        except Exception as e:

            writeLog("BayesianSearch: Testing config: {} - Exception {}\n".format(paramether_dictionary, str(e)),
                     self.logFile)
            traceback.print_exc()

            return - np.inf


def function_interface(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1


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
            W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

        if verbose:
            print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W

    else:
        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

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
        W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
        W_sparse = W_sparse.tocsr()

        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse


def areURMequals(URM1, URM2):
    if (URM1.shape != URM2.shape):
        return False

    return (URM1 - URM2).nnz == 0


def removeTopPop(URM_1, URM_2=None, percentageToRemove=0.2):
    """
    Remove the top popular items from the matrix
    :param URM_1: user X items
    :param URM_2: user X items
    :param percentageToRemove: value 1 corresponds to 100%
    :return: URM: user X selectedItems, obtained from URM_1
             Array: itemMappings[selectedItemIndex] = originalItemIndex
             Array: removedItems
    """

    item_pop = URM_1.sum(axis=0)  # this command returns a numpy.matrix of size (1, nitems)

    if URM_2 != None:
        assert URM_2.shape[1] == URM_1.shape[1], \
            "The two URM do not contain the same number of columns, URM_1 has {}, URM_2 has {}".format(URM_1.shape[1],
                                                                                                       URM_2.shape[1])

        item_pop += URM_2.sum(axis=0)

    item_pop = np.asarray(item_pop).squeeze()  # necessary to convert it into a numpy.array of size (nitems,)
    popularItemsSorted = np.argsort(item_pop)[::-1]

    numItemsToRemove = int(len(popularItemsSorted) * percentageToRemove)

    # Choose which columns to keep
    itemMask = np.in1d(np.arange(len(popularItemsSorted)), popularItemsSorted[:numItemsToRemove], invert=True)

    # Map the column index of the new URM to the original ItemID
    itemMappings = np.arange(len(popularItemsSorted))[itemMask]

    removedItems = np.arange(len(popularItemsSorted))[np.logical_not(itemMask)]

    return URM_1[:, itemMask], itemMappings, removedItems


#
#
# def load_edges (filePath, header = False):
#
#     values, rows, cols = [], [], []
#
#     fileHandle = open(filePath, "r")
#     numCells = 0
#
#     if header:
#         fileHandle.readline()
#
#     for line in fileHandle:
#         numCells += 1
#         if (numCells % 1000000 == 0):
#             print("Processed {} cells".format(numCells))
#
#         if (len(line)) > 1:
#             line = line.split(",")
#
#             value = line[2].replace("\n", "")
#
#             if not value == "0" and not value == "NaN":
#                 rows.append(int(line[0]))
#                 cols.append(int(line[1]))
#                 values.append(float(value))
#
#     return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)


def addZeroSamples(S_matrix, numSamplesToAdd):
    n_items = S_matrix.shape[1]

    S_matrix_coo = S_matrix.tocoo()

    row_index = list(S_matrix_coo.row)
    col_index = list(S_matrix_coo.col)
    data = list(S_matrix_coo.data)

    existingSamples = set(zip(row_index, col_index))

    addedSamples = 0
    consecutiveFailures = 0

    while (addedSamples < numSamplesToAdd):

        item1 = np.random.randint(0, n_items)
        item2 = np.random.randint(0, n_items)

        if (item1 != item2 and (item1, item2) not in existingSamples):

            row_index.append(item1)
            col_index.append(item2)
            data.append(0)

            existingSamples.add((item1, item2))

            addedSamples += 1
            consecutiveFailures = 0

        else:
            consecutiveFailures += 1

        if (consecutiveFailures >= 100):
            raise SystemExit(
                "Unable to generate required zero samples, termination at 100 consecutive discarded samples")

    return row_index, col_index, data


def reshapeSparse(sparseMatrix, newShape):
    if sparseMatrix.shape[0] > newShape[0] or sparseMatrix.shape[1] > newShape[1]:
        ValueError("New shape cannot be smaller than SparseMatrix. SparseMatrix shape is: {}, newShape is {}".format(
            sparseMatrix.shape, newShape))

    sparseMatrix = sparseMatrix.tocoo()
    newMatrix = sps.csr_matrix((sparseMatrix.data, (sparseMatrix.row, sparseMatrix.col)), shape=newShape)

    return newMatrix


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


def compute_PageRank(G, beta=0.85, epsilon=10 ** -4):
    '''
    Efficient computation of the PageRank values using a sparse adjacency
    matrix and the iterative power method.

    Parameters
    ----------
    G : boolean adjacency matrix. np.bool8
        If the element j,i is True, means that there is a link from i to j.
    beta: 1-teleportation probability.
    epsilon: stop condition. Minimum allowed amount of change in the PageRanks
        between iterations.

    Returns
    -------
    output : tuple
        PageRank array normalized top one.
        Number of iterations.

    '''
    # Test adjacency matrix is OK
    n, _ = G.shape
    assert (G.shape == (n, n))
    # Constants Speed-UP
    deg_out_beta = G.sum(axis=0).T / beta  # vector
    # Initialize
    ranks = np.ones((n, 1)) / n  # vector
    time = 0
    flag = True
    while flag:
        time += 1
        with np.errstate(divide='ignore'):  # Ignore division by 0 on ranks/deg_out_beta
            new_ranks = G.dot((ranks / deg_out_beta))  # vector
        # Leaked PageRank
        new_ranks += (1 - new_ranks.sum()) / n
        # Stop condition
        if np.linalg.norm(ranks - new_ranks, ord=1) <= epsilon:
            flag = False
        ranks = new_ranks
    return ranks / ranks.max()


def get_fake_test():
    return sparse.load_npz(os.path.join("Dataset", "fake_test.npz"))


def divide_train_test(train_old, threshold=0.8):
    msk = np.random.rand(len(train_old)) < threshold
    train = train_old[msk]
    test = train_old[~msk]
    return train, test


def add_dataframe(df, playlist_id, songs_list):
    if type(playlist_id) is list:
        data = pd.DataFrame({"playlist_id": playlist_id, "track_id": songs_list})
    else:
        data = pd.DataFrame({"playlist_id": [playlist_id] * len(songs_list), "track_id": songs_list})
    df = df.append(data)
    return df


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
        self.train = pd.read_csv(os.path.join("Dataset", "train.csv"))
        self.train_sequential = pd.read_csv(os.path.join("Dataset", "train_sequential.csv"))
        self.tracks = pd.read_csv(os.path.join("Dataset", "tracks.csv"))
        self.target_playlist = pd.read_csv(os.path.join("Dataset", "target_playlists.csv"))
        self.ICM = None

        if all_train:
            self.UCB_tfidf_artists = self.get_UCM_matrix_artists(train_path=os.path.join("Dataset", "train.csv"))
            # self.UCB_tfidf_album = self.get_UCM_matrix_album(train_path=os.path.join("Dataset", "train.csv"))
            self.URM_train = create_URM_matrix(self.train)
            self.URM_test = get_fake_test()
            self.URM_validation = get_fake_test()
        else:
            if self.distr_split:
                try:
                    self.UCB_tfidf_artists = self.get_UCM_matrix_artists(train_path=os.path.join("Dataset",
                                                                                                 "train_divided_keep_distrib.csv"))
                    # self.UCB_tfidf_album = self.get_UCM_matrix_album(train_path=os.path.join("Dataset", "new_train.csv"))
                    self.URM_train = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_train_keep_distrib.npz"))
                    self.URM_test = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_test_keep_distrib.npz"))
                    self.URM_validation = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_test_keep_distrib.npz"))
                except FileNotFoundError:
                    data_grouped = self.train_sequential.groupby(self.train_sequential.playlist_id).track_id.apply(list)
                    train_keep_dist = pd.DataFrame(columns=["playlist_id", "track_id"])
                    test_keep_dist = pd.DataFrame(columns=["playlist_id", "track_id"])

                    for i in data_grouped.keys():
                        line = data_grouped[i]
                        len20 = int(len(line) * .8)
                        train_keep_dist = add_dataframe(train_keep_dist, i, line[:len20])
                        test_keep_dist = add_dataframe(test_keep_dist, i, line[len20:])

                    sequential_playlists = data_grouped.keys()
                    data_gr_all = self.train.groupby(self.train.playlist_id).track_id.apply(list)

                    to_add_train, to_add_test = [], []
                    to_add_train_ind, to_add_test_ind = [], []
                    for i in data_gr_all.keys():
                        if i not in sequential_playlists:
                            line = data_gr_all[i]
                            len20 = int(len(line) * .8)
                            indexes = random.sample(range(0, len(line)), len20)
                            for ind, el in enumerate(line):
                                if ind in indexes:
                                    to_add_train_ind.append(i)
                                    to_add_train.append(el)
                                else:
                                    to_add_test_ind.append(i)
                                    to_add_test.append(el)
                    train_keep_dist = add_dataframe(train_keep_dist, to_add_train_ind, to_add_train)
                    test_keep_dist = add_dataframe(test_keep_dist, to_add_test_ind, to_add_test)

                    self.URM_train = create_URM_matrix(train_keep_dist)
                    self.URM_test = create_URM_matrix(test_keep_dist)
                    self.URM_validation = create_URM_matrix(test_keep_dist)

                    sparse.save_npz(os.path.join("IntermediateComputations", "URM_train_keep_distrib.npz"),
                                    self.URM_train)
                    sparse.save_npz(os.path.join("IntermediateComputations", "URM_test_keep_distrib.npz"),
                                    self.URM_test)
                    train_keep_dist.to_csv(os.path.join("Dataset", "train_divided_keep_distrib.csv"))
            elif self.top10k:
                try:
                    self.UCB_tfidf_artists = self.get_UCM_matrix_artists(train_path=os.path.join("Dataset",
                                                                                                 "new_train.csv"))
                    # self.UCB_tfidf_album = self.get_UCM_matrix_album(train_path=os.path.join("Dataset", "new_train.csv"))
                    self.URM_train = scipy.sparse.load_npz(os.path.join("IntermediateComputations", "URM_train.npz"))
                    self.URM_test = scipy.sparse.load_npz(os.path.join("IntermediateComputations", "URM_test.npz"))
                    self.URM_validation = scipy.sparse.load_npz(
                        os.path.join("IntermediateComputations", "URM_test.npz"))
                except FileNotFoundError:
                    start_mask = np.asarray([False] * len(self.train))
                    with open(os.path.join("Dataset", 'top10_playlist.pkl'), 'rb') as handle:
                        top10k_playlist = pickle.load(handle)

                    for top_play in top10k_playlist:
                        my_train = self.train[self.train.playlist_id == top_play]
                        to_take = random.sample(list(my_train.index), 10)
                        start_mask[to_take] = True

                    new_train = self.train[~start_mask]
                    new_test = self.train[start_mask]

                    self.URM_train = create_URM_matrix(new_train)
                    self.URM_test = create_URM_matrix(new_test)
                    self.URM_validation = create_URM_matrix(new_test)

                    new_train.to_csv(os.path.join("Dataset", "new_train.csv"))
                    scipy.sparse.save_npz(os.path.join("IntermediateComputations", "URM_train.npz"), self.URM_train)
                    scipy.sparse.save_npz(os.path.join("IntermediateComputations", "URM_test.npz"), self.URM_test)
                    self.UCB_tfidf_artists = self.get_UCM_matrix_artists(train_path=os.path.join("Dataset",
                                                                                                 "new_train.csv"))
                    # here we use the same train and test

            else:
                train, test = divide_train_test(self.train, threshold=0.85)

                self.URM_train = create_URM_matrix(train)
                self.URM_test = create_URM_matrix(test)
                self.URM_validation = self.URM_test
                train.to_csv(os.path.join("Dataset", "new_train_no_top10k.csv"))
                self.UCB_tfidf_artists = self.get_UCM_matrix_artists(
                    train_path=os.path.join("Dataset", "new_train_no_top10k.csv"))
                # URM_all = create_URM_matrix(train)
                # self.URM_train, self.URM_test, self.URM_validation = split_train_validation_test(URM_all,
                #                                                                                  split_train_test_validation_quota)

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

    def get_ICM(self):
        if self.ICM is None:
            self.ICM = get_icm_matrix(self.tracks)
        return self.ICM

    def get_page_rank_URM(self):
        try:
            with open(os.path.join("Dataset", "URM_pagerank.pkl"), 'rb') as handle:
                to_ret = pickle.load(handle)
                # print(to_ret)
                return to_ret
        except FileNotFoundError:
            self.URM_train = self.URM_train.astype(float)
            l = range(self.URM_train.shape[1])
            s_all = set(l)
            relation_mat_gen = self.URM_train.transpose().dot(self.URM_train).tocsc()
            t = time.time()
            URM_new = self.URM_train
            for user_id in range(self.URM_train.shape[0]):
                if user_id % 100 == 0:
                    print(user_id)
                    print("Avg time spent: {}".format((time.time() - t) / 100))
                    t = time.time()
                relation_mat = relation_mat_gen.copy()
                songs_in_playlist = self.URM_train.indices[
                                    self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                s_i = s_all - set(songs_in_playlist)
                for i in s_i:
                    relation_mat.data[relation_mat.indptr[i]:relation_mat.indptr[i + 1]].fill(0)
                relation_mat.eliminate_zeros()
                page_rank = compute_PageRank(relation_mat.transpose()).A1
                # print(type(self.URM_train[user_id].data))
                # URM_new[user_id].data = np.multiply(URM_new[user_id].data, page_rank[songs_in_playlist])
                URM_new[user_id] = URM_new[user_id].multiply(page_rank)
                del relation_mat
            print("URM modified")
            with open(os.path.join("Dataset", "URM_pagerank.pkl"), 'wb') as handle:
                pickle.dump(URM_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return self.URM_train

    def create_complete_test(self):
        row = 50446
        col = 20635
        return csr_matrix(([1] * row, (range(row), [0] * row)), shape=(row, col))

    def get_UCM_matrix_artists(self, train_path=""):
        return None


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

    def saveModel(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"sparse_weights": self.sparse_weights}

        if self.sparse_weights:
            try:
                dictionary_to_save["W_sparse"] = self.W_sparse
            except AttributeError:
                print("I'm an hybrid recommender, so I don't have a similarity to save!")

        else:
            dictionary_to_save["W"] = self.W

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))


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

    def saveModel(self, folder_path, file_name=None):
        raise NotImplementedError("Recommender: saveModel not implemented")

    def loadModel(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Loading model from file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = pickle.load(open(folder_path + file_name, "rb"))

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        print("{}: Loading complete".format(self.RECOMMENDER_NAME))


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

    def fit(self, alpha=1., beta=0.6, min_rating=0, topK=100, implicit=True, normalize_similarity=True,
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
                with open(os.path.join("IntermediateComputations", "RP3betaMatrix.pkl"), 'rb') as handle:
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

        with open(os.path.join("IntermediateComputations", "RP3betaMatrix.pkl"), 'wb') as handle:
            pickle.dump((self.topK, self.W_sparse), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("RP3beta similarity matrix saved")


def runParameterSearch_Collaborative(recommender_class, URM_train, ICM=None, metric_to_optimize="MAP",
                                     evaluator_validation=None, evaluator_test=None,
                                     evaluator_validation_earlystopping=None,
                                     output_root_path="result_experiments/", parallelizeKNN=False, n_cases=30,
                                     URM_validation=None, UCM_train=None):
    from ParameterTuning.AbstractClassSearch import DictionaryKeys

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    try:

        output_root_path_rec_name = output_root_path + recommender_class.RECOMMENDER_NAME

        parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation)

        if recommender_class is RP3betaRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = list(range(1, 800, 5))
            hyperparamethers_range_dictionary["alpha"] = list(np.linspace(0.001, 2.0, 500))  # range(0, 2)
            hyperparamethers_range_dictionary["beta"] = list(np.linspace(0.001, 2.0, 500))  # range(0, 2) np.linespace()
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################


        # if recommender_class is MultiThreadSLIM_ElasticNet:
        #     hyperparamethers_range_dictionary = {}
        #     hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
        #     hyperparamethers_range_dictionary["l1_penalty"] = [1.0, 0.1, 1e-2, 1e-4, 1e-6]
        #     hyperparamethers_range_dictionary["l2_penalty"] = [1.0, 0.1, 1e-2, 1e-4, 1e-6]
        #
        #     recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
        #                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
        #                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
        #                              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
        #                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 n_cases=n_cases,
                                                 output_root_path=output_root_path_rec_name,
                                                 metric=metric_to_optimize,
                                                 init_points=20
                                                 )



    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))

        error_file = open(output_root_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    # this line removes old matrices saved, comment it if testing only the weights of hybrid
    # delete_previous_intermediate_computations()
    dataReader = RS_Data_Loader()

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    UCM_train = dataReader.get_tfidf_artists()
    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    print("dataset loaded")

    collaborative_algorithm_list = [
        # Random,
        # TopPop,
        # ItemKNNCBFRecommender,
        # UserKNNCBRecommender,
        # P3alphaRecommender,
        RP3betaRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender,
        # MultiThreadSLIM_ElasticNet,
        # HybridRecommender
    ]

    # if UserKNNCBRecommender in collaborative_algorithm_list:
    #     ICM = dataReader.get_tfidf_artists()
    #     ICM = dataReader.get_tfidf_album()
    # elif ItemKNNCBFRecommender in collaborative_algorithm_list:
    #     ICM = dataReader.get_ICM()


    evaluator_validation_earlystopping = SequentialEvaluator(URM_validation, URM_train, cutoff_list=[10])
    evaluator_test = SequentialEvaluator(URM_test, URM_train, cutoff_list=[10])

    evaluator_validation = EvaluatorWrapper(evaluator_validation_earlystopping)
    evaluator_test = EvaluatorWrapper(evaluator_test)

    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                               URM_train=URM_train,
                                                               ICM=ICM,
                                                               metric_to_optimize="MAP",
                                                               evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                               evaluator_validation=evaluator_validation,
                                                               # evaluator_test=evaluator_test, # I'm not
                                                               # passing it because validation and test for
                                                               # us is the same
                                                               output_root_path=output_root_path,
                                                               n_cases=30,
                                                               URM_validation=URM_validation)
            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))

        print("ciao")


if __name__ == '__main__':
    read_data_split_and_search()
