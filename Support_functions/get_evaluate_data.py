import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os, pickle
from itertools import repeat


def divide_train_test(train_old, threshold=0.8):
    msk = np.random.rand(len(train_old)) < threshold
    train = train_old[msk]
    test = train_old[~msk]
    return train, test


def make_series(test):
    dict_train = test.groupby(test.playlist_id).track_id
    test_songs_per_playlist = pd.Series(dict_train.apply(list))
    return test_songs_per_playlist


def tracks_popularity():
    with open(os.path.join("IntermediateComputations", "dic_pop.pkl_no_rem"), 'rb') as handle:
        to_ret = pickle.load(handle)
    return to_ret
    # all_train, train, test, tracks, target_playlist, all_playlist_to_predict, test_songs_per_playlist, validation = get_data()
    # scores = all_train.groupby('track_id').count()
    # scores.columns = ['scores']
    # to_ret = scores.to_dict()['scores']
    # with open(os.path.join("IntermediateComputations", "dic_pop.pkl_no_rem"), 'wb') as handle:
    #     pickle.dump(to_ret, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # return


def fill_URM_with_reccomendations(URM, target_recommendations):
    URM_lil = URM.tolil()
    for recommendations in target_recommendations:
        URM[recommendations[0], recommendations[1][:5]] = 1
        print(recommendations[0])
    return csr_matrix(URM_lil)


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


def get_data():
    train = pd.read_csv(os.path.join("Dataset", "train.csv"))
    all_playlist_to_predict = pd.DataFrame(index=train.playlist_id.unique())
    s = train.playlist_id.unique()
    tracks = pd.read_csv(os.path.join("Dataset", "tracks.csv"))
    target_playlist = pd.read_csv(os.path.join("Dataset", "target_playlists.csv"))
    all_train = train.copy()
    train, test = divide_train_test(train, threshold=0.85)
    train, validation = divide_train_test(train, threshold=0.85)
    test_songs_per_playlist = make_series(test)
    all_playlist_to_predict['playlist_id'] = pd.Series(range(len(all_playlist_to_predict.index)),
                                                       index=all_playlist_to_predict.index)
    return all_train, train, test, tracks, target_playlist, all_playlist_to_predict, test_songs_per_playlist, validation


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    # noinspection PyUnresolvedReferences
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    # noinspection PyUnresolvedReferences
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


# all_playlist_to_predict must be the same as the submission version, just with the recommendations for all playlist
# test_songs_per_playlist could be untouched from the beginning
def evaluate_algorithm(all_playlist_to_predict, test_songs_per_playlist, users_number=50445):
    if 'playlist_id' in all_playlist_to_predict.columns:
        tmp = all_playlist_to_predict.copy()
        tmp.set_index('playlist_id', inplace=True)
        train_series = tmp.ix[:, 0]

    else:
        train_series = all_playlist_to_predict.ix[:, 0]

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for user_id in range(users_number + 1):
        if user_id in test_songs_per_playlist.index:
            relevant_items = np.asarray(test_songs_per_playlist[user_id])

            recommended_items = train_series[user_id]
            # recommended_items = [int(x) for x in recommended_items.split(" ")]
            recommended_items = np.asarray(recommended_items)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    to_print = "Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP)
    print(to_print)
    return to_print


def create_URM_matrix(train):
    row = list(train.playlist_id)
    col = list(train.track_id)
    return csr_matrix(([1] * len(row), (row, col)), shape=(50446, 20635))
