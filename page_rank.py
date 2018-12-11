import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix

from Dataset.RS_Data_Loader import RS_Data_Loader


def pageRank(G, s=.85, maxerr=.0001, non_zero_columns=None):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
       Gij is a binary value representing a transition from state i to j.
    s: probability of following a transition. 1-s probability of teleporting
       to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
    """
    n = G.shape[0]

    # transform G into markov matrix A
    if G is not csc_matrix:
        print("Non è csc")
        A = csc_matrix(G, dtype=np.float)
    else:
        A = G

    if non_zero_columns is None:
        non_zero_columns = range(n)
    rsums = np.array(A.sum(1))[:, 0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

    # bool array of sink states
    sink = rsums == 0

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    # r[non_zero_columns] = 1
    while np.sum(np.abs(r - ro)) > maxerr:
        print("ciclo while error: {}".format(np.sum(np.abs(r - ro))))
        ro = r.copy()
        # calculate each pagerank at a time
        for i in non_zero_columns:
            # inlinks of state i
            Ai = np.array(A[:, i].todense())[:, 0]
            # account for sink states
            Di = sink / float(n)

            # account for teleportation to state i
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot(Ai * s + Di * s + Ei * (1 - s))
            if np.inf in Ai * s + Di * s + Ei * (1 - s) or np.inf in r:
                print("Inf in product")
            if r[i] == np.inf:
                print("questo è inf: {}, {}".format(r[i], Ai * s + Di * s + Ei * (1 - s)))

    # return normalized pagerank
    print("final divide:", float(sum(r)))
    return r / float(sum(r))


if __name__ == '__main__':

    dataReader = RS_Data_Loader()

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    relation_mat = URM_train.transpose().dot(URM_train).tocsc()
    user_id = 0
    songs_in_playlist = URM_train.indices[URM_train.indptr[user_id]:URM_train.indptr[user_id + 1]]
    l = range(URM_train.shape[1])
    s_all = set(l)
    s_0 = s_all - set(songs_in_playlist)

    for i in s_0:
        relation_mat.data[relation_mat.indptr[i]:relation_mat.indptr[i + 1]].fill(0)
    relation_mat.eliminate_zeros()

    G = nx.from_numpy_matrix(relation_mat.A)
    pr = nx.pagerank(G, alpha=0.9)
    # p_r_vector = powerIteration(relation_mat, rsp=0.15, epsilon=0.00001, maxIterations=1000)
    print(pr)
    pr_array = np.array(pr.values())
    print(pr_array[songs_in_playlist])
