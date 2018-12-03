from scipy.sparse import csr_matrix

from Dataset.RS_Data_Loader import RS_Data_Loader

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from scipy.sparse.csgraph import connected_components
import traceback, os
import numpy as np
import hdbscan as clust

if __name__ == '__main__':
    evaluate_algorithm = True
    slim_after_hybrid = False

    # delete_previous_intermediate_computations()

    filename = "hybrid_UserContentMatrix"

    dataReader = RS_Data_Loader(top10k=True, all_train=not evaluate_algorithm)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()

    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, URM_train, exclude_seen=True)

    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logFile = open(output_root_path + "result_all_algorithms.txt", "a")

    recommender_class = UserKNNCFRecommender
    print("Algorithm: {}".format(recommender_class))
    recommender = recommender_class(URM_train)

    recommender.fit(topK=200, shrink=10, similarity='cosine', normalize=True, force_compute_sim=True)




    def delete_row_csr(mat, i):

        n = mat.indptr[i + 1] - mat.indptr[i]
        if n > 0:
            mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i + 1]:]
            mat.data = mat.data[:-n]
            mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i + 1]:]
            mat.indices = mat.indices[:-n]
        mat.indptr[i:-1] = mat.indptr[i + 1:]
        mat.indptr[i:] -= n
        mat.indptr = mat.indptr[:-1]
        mat._shape = (mat._shape[0] - 1, mat._shape[1])


    def dropcols_coo(M, idx_to_drop):
        idx_to_drop = np.unique(idx_to_drop)
        C = M.tocoo()
        keep = ~np.in1d(C.col, idx_to_drop)
        C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
        C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
        C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
        return C.tocsr()


    print(recommender.W_sparse.shape)

    while True:
        connected_c = connected_components(recommender.W_sparse)
        index = 0

        print(connected_c)
        for x in connected_c[1]:
            if x != 0:
                delete_row_csr(URM_train, index)
                print(index)
            index += 1

        recommender = recommender_class(csr_matrix(URM_train))
        recommender.fit(topK=200, shrink=10, similarity='cosine', normalize=True, force_compute_sim=True)
        print(recommender.W_sparse.shape)

        try:
            clusterer = clust.HDBSCAN(metric='precomputed')
            clusterer.fit(recommender.W_sparse)
            print(clusterer.labels_)
            break
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    print(recommender.W_sparse.shape)


    # print("Creating clusterer...")
    # clusterer = clust.HDBSCAN(metric='precomputed')
    # print("Clusterizing...")
    # clusterer.fit(recommender.W_sparse)
    #
    print(clusterer.labels_)