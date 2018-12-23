#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/06/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
import time, sys, copy
import matplotlib.pyplot as plt
from enum import Enum
import Support_functions.get_evaluate_data as ged
from Base.Evaluation.metrics import roc_auc, precision, recall, recall_min_test_len, map, ndcg, rr, arhr, \
    Novelty, Coverage_Item, Metrics_Object, Coverage_User, Gini_Diversity, Shannon_Entropy, Diversity_MeanInterList, \
    Diversity_Herfindahl


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


import scipy.sparse as sps


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


def plot(data_stats):
    label = []
    for i in data_stats:
        content = data_stats[i]
        lenght = len(content)
        summed_values = sum(content)
        data_stats[i] = summed_values / float(lenght)
        label.append(str(lenght))

    z, y = list(data_stats.keys()), list(data_stats.values())
    fig, ax = plt.subplots()
    ax.scatter(z, y)

    for i, txt in enumerate(label):
        ax.annotate(txt, (z[i], y[i]))
    fig.show()


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

        dict_song_pop = ged.tracks_popularity()
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
                                                                        remove_CustomItems_flag=self.ignore_items_flag,
                                                                        dict_pop=dict_song_pop)

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
                if plot_stats:

                    key_pop = int(ged.playlist_popularity(user_profile, pop_dict=dict_song_pop))
                    key_len = int(ged.lenght_playlist(user_profile))

                    if key_pop not in data_stats_pop:
                        data_stats_pop[key_pop] = [current_map]
                    else:
                        data_stats_pop[key_pop].append(current_map)

                    if key_len not in data_stats_len:
                        data_stats_len[key_len] = [current_map]
                    else:
                        data_stats_len[key_len].append(current_map)

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

        if plot_stats:
            if onPop:
                plot(data_stats_pop)
            else:
                plot(data_stats_len)

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


class ParallelEvaluator(Evaluator):
    """ParallelEvaluator"""

    EVALUATOR_NAME = "ParallelEvaluator_Class"

    def __init__(self, URM_test_list, cutoff_list, minRatingsPerUser=1, exclude_seen=True,
                 diversity_object=None,
                 ignore_items=None,
                 ignore_users=None):

        super(ParallelEvaluator, self).__init__(URM_test_list, cutoff_list,
                                                diversity_object=diversity_object,
                                                minRatingsPerUser=minRatingsPerUser, exclude_seen=exclude_seen,
                                                ignore_items=ignore_items, ignore_users=ignore_users)

    def evaluateRecommender(self, recommender_object, n_processes=None):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        if n_processes is None:
            n_processes = int(multiprocessing.cpu_count() / 2)

        start_time = time.time()

        # Split the users to evaluate
        n_processes = min(n_processes, len(self.usersToEvaluate))
        batch_len = int(len(self.usersToEvaluate) / n_processes)
        batch_len = max(batch_len, 1)

        sequential_evaluators_list = []
        sequential_evaluators_n_users_list = []

        for n_evaluator in range(n_processes):

            stat_user = n_evaluator * batch_len
            end_user = min((n_evaluator + 1) * batch_len, len(self.usersToEvaluate))

            if n_evaluator == n_processes - 1:
                end_user = len(self.usersToEvaluate)

            batch_users = self.usersToEvaluate[stat_user:end_user]
            sequential_evaluators_n_users_list.append(len(batch_users))

            not_in_batch_users = np.in1d(self.usersToEvaluate, batch_users, invert=True)
            not_in_batch_users = np.array(self.usersToEvaluate)[not_in_batch_users]

            new_evaluator = _ParallelEvaluator_batch(self.URM_test, self.cutoff_list, ignore_users=not_in_batch_users)

            sequential_evaluators_list.append(new_evaluator)

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        run_parallel_evaluator_partial = partial(_run_parallel_evaluator, recommender_object=recommender_object)

        pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=1)
        resultList = pool.map(run_parallel_evaluator_partial, sequential_evaluators_list)

        print("ParallelEvaluator: Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
            len(self.usersToEvaluate),
            100.0 * float(len(self.usersToEvaluate)) / len(self.usersToEvaluate),
            time.time() - start_time,
            float(len(self.usersToEvaluate)) / (time.time() - start_time)))

        sys.stdout.flush()
        sys.stderr.flush()

        results_dict = {}
        n_users_evaluated = 0

        for cutoff in self.cutoff_list:
            results_dict[cutoff] = create_empty_metrics_dict(self.n_items, self.n_users,
                                                             recommender_object.URM_train,
                                                             self.ignore_items_ID,
                                                             self.ignore_users_ID,
                                                             cutoff,
                                                             self.diversity_object)

        for new_result_index in range(len(resultList)):
            print("Result list: {}".format(resultList[new_result_index]))
            new_result, n_users_evaluated_batch = resultList[new_result_index]
            n_users_evaluated += n_users_evaluated_batch

            results_dict = _merge_results_dict(results_dict, new_result, n_users_evaluated_batch)

        for cutoff in self.cutoff_list:
            for key in results_dict[cutoff].keys():
                results_dict[cutoff][key] /= len(self.usersToEvaluate)

        if n_users_evaluated > 0:

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

        sequential_evaluators_list = None
        sequential_evaluators_n_users_list = None

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        results_run_string = self.get_result_string(results_dict)

        return (results_dict, results_run_string)


class LeaveOneOutEvaluator(Evaluator):
    """SequentialEvaluator"""

    EVALUATOR_NAME = "LeaveOneOutEvaluator_Class"

    def __init__(self, URM_test_list, URM_test_negative, cutoff_list, minRatingsPerUser=1, exclude_seen=True,
                 diversity_object=None,
                 ignore_items=None,
                 ignore_users=None):
        """

        :param URM_test_list:
        :param URM_test_negative: Items to rank together with the test items
        :param cutoff_list:
        :param minRatingsPerUser:
        :param exclude_seen:
        :param diversity_object:
        :param ignore_items:
        :param ignore_users:
        """

        super(LeaveOneOutEvaluator, self).__init__(URM_test_list, cutoff_list,
                                                   diversity_object=diversity_object,
                                                   minRatingsPerUser=minRatingsPerUser, exclude_seen=exclude_seen,
                                                   ignore_items=ignore_items, ignore_users=ignore_users)

        self.URM_test_negative = sps.csr_matrix(URM_test_negative)

    def user_specific_remove_items(self, recommender_object, user_id):

        self.ignore_items_flag = True

        self._global_ignore_items_ID = self.ignore_items_ID.copy()

        # items_to_remove_for_user = self.__all_items.copy()
        items_to_remove_for_user_mask = self.__all_items_mask.copy()

        ### ADD negative samples
        start_pos = self.URM_test_negative.indptr[user_id]
        end_pos = self.URM_test_negative.indptr[user_id + 1]

        items_to_remove_for_user_mask[self.URM_test_negative.indices[start_pos:end_pos]] = False

        ### ADD positive samples
        start_pos = self.URM_test.indptr[user_id]
        end_pos = self.URM_test.indptr[user_id + 1]

        items_to_remove_for_user_mask[self.URM_test.indices[start_pos:end_pos]] = False

        recommender_object.set_items_to_ignore(self.__all_items[items_to_remove_for_user_mask])

    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        results_dict = {}

        for cutoff in self.cutoff_list:
            results_dict[cutoff] = create_empty_metrics_dict(self.n_items, self.n_users,
                                                             recommender_object.URM_train,
                                                             self.ignore_items_ID,
                                                             self.ignore_users_ID,
                                                             cutoff,
                                                             self.diversity_object)

        start_time = time.time()
        start_time_print = time.time()

        n_eval = 0

        self.__all_items = np.arange(0, self.n_items, dtype=np.int)
        self.__all_items = set(self.__all_items)

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        for test_user in self.usersToEvaluate:

            # Being the URM CSR, the indices are the non-zero column indexes
            relevant_items = self.get_user_relevant_items(test_user)

            n_eval += 1

            self.user_specific_remove_items(recommender_object, test_user)

            # recommended_items = recommender_object.recommend(np.array(test_user), remove_seen_flag=self.exclude_seen,
            #                                                  cutoff = self.max_cutoff, remove_top_pop_flag=False, remove_CustomItems_flag=self.ignore_items_flag)
            recommended_items = recommender_object.recommend(np.atleast_1d(test_user),
                                                             remove_seen_flag=self.exclude_seen,
                                                             cutoff=self.max_cutoff,
                                                             remove_top_pop_flag=False,
                                                             remove_CustomItems_flag=self.ignore_items_flag)

            recommended_items = np.array(recommended_items[0])

            recommender_object.reset_items_to_ignore()

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

            if time.time() - start_time_print > 30 or n_eval == len(self.usersToEvaluate):
                print(
                    "SequentialEvaluator: Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                        n_eval,
                        100.0 * float(n_eval) / len(self.usersToEvaluate),
                        time.time() - start_time,
                        float(n_eval) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print = time.time()

        if (n_eval > 0):

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                for key in results_current_cutoff.keys():

                    value = results_current_cutoff[key]

                    if isinstance(value, Metrics_Object):
                        results_current_cutoff[key] = value.get_metric_value()
                    else:
                        results_current_cutoff[key] = value / n_eval

                precision_ = results_current_cutoff[EvaluatorMetrics.PRECISION.value]
                recall_ = results_current_cutoff[EvaluatorMetrics.RECALL.value]

                if precision_ + recall_ != 0:
                    results_current_cutoff[EvaluatorMetrics.F1.value] = 2 * (precision_ * recall_) / (
                        precision_ + recall_)


        else:
            print("WARNING: No users had a sufficient number of relevant items")

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        results_run_string = self.get_result_string(results_dict)

        return (results_dict, results_run_string)
