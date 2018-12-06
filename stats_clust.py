import pickle

from scipy.sparse import csr_matrix

from Dataset.RS_Data_Loader import RS_Data_Loader

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from scipy.sparse.csgraph import connected_components
import traceback, os
import numpy as np
from sklearn.cluster import KMeans

if __name__ == '__main__':
    evaluate_algorithm = True
    delete_old_computations = False
    slim_after_hybrid = False

    # delete_previous_intermediate_computations()
    # if not evaluate_algorithm:
        # delete_previous_intermediate_computations()
    # else:
    # print("ATTENTION: old intermediate computations kept, pay attention if running with all_train")

    filename = "best_hybrid_new_dataset.csv"

    dataReader = RS_Data_Loader(all_train=not evaluate_algorithm)

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    UCM_tfidf = dataReader.get_tfidf_artists()

    with open(os.path.join("IntermediateComputations", "Clusterization_Kmeans_3.pkl"), 'rb') as handle:
        indices = pickle.load(handle)
        print(indices)

    with open(os.path.join("IntermediateComputations", "Cluster_0_Kmeans_3.pkl"), 'rb') as handle:
        cluster_0 = pickle.load(handle)

    with open(os.path.join("IntermediateComputations", "Cluster_1_Kmeans_3.pkl"), 'rb') as handle:
        cluster_1 = pickle.load(handle)

    with open(os.path.join("IntermediateComputations", "Cluster_2_Kmeans_3.pkl"), 'rb') as handle:
        cluster_2 = pickle.load(handle)

    indices_0 = []
    indices_1 = []
    indices_2 = []
    for index in range(indices.shape[0]):
        if indices[index] == 0:
            indices_0.append(index)
        elif indices[index] == 1:
            indices_1.append(index)
        elif indices[index] == 2:
            indices_2.append(index)

    dict_cluster_0 = {}
    dict_cluster_1 = {}
    dict_cluster_2 = {}

    dict_cluster_0.fromkeys(indices_0, range(cluster_0.shape[0]))
    dict_cluster_1.fromkeys(indices_1, range(cluster_1.shape[0]))
    dict_cluster_2.fromkeys(indices_2, range(cluster_2.shape[0]))

    print(indices_0)
    print(indices_1)
    print(indices_2)

    with open(os.path.join("IntermediateComputations", "Cluster_0_dict_Kmeans_3.pkl"), 'wb') as handle:
        pickle.dump(indices_0, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("IntermediateComputations", "Cluster_1_dict_Kmeans_3.pkl"), 'wb') as handle:
        pickle.dump(indices_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("IntermediateComputations", "Cluster_2_dict_Kmeans_3.pkl"), 'wb') as handle:
        pickle.dump(indices_2, handle, protocol=pickle.HIGHEST_PROTOCOL)


