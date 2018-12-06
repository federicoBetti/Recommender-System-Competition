import pickle

from scipy.sparse import csr_matrix

from Dataset.RS_Data_Loader import RS_Data_Loader

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from scipy.sparse.csgraph import connected_components
import traceback, os
import numpy as np
import hdbscan as clust
from sklearn.cluster import KMeans

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


    with open(os.path.join("IntermediateComputations", "Clusterization_Kmeans_3.pkl"), 'rb') as handle:
        clusters = pickle.load(handle)

    cl_0_ind = []
    cl_1_ind = []
    cl_2_ind = []
    for elem in clusters:
        if elem == 0:
            cl_0_ind.append(elem)
        elif elem == 1:
            cl_1_ind.append(elem)
        else:
            cl_2_ind.append(elem)

    cluster_0 = URM_train[cl_0_ind]
    print(cluster_0.shape)
    cluster_1 = URM_train[cl_1_ind]
    cluster_2 = URM_train[cl_2_ind]

    with open(os.path.join("IntermediateComputations", "Cluster_0_Kmeans_3.pkl"), 'wb') as handle:
        pickle.dump(cluster_0, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("IntermediateComputations", "Cluster_1_Kmeans_3.pkl"), 'wb') as handle:
        pickle.dump(cluster_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("IntermediateComputations", "Cluster_2_Kmeans_3.pkl"), 'wb') as handle:
        pickle.dump(cluster_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
