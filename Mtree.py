from BaseIndex import BaseIndex
from cophir_distances import get_cophir_distance
from imports import np
class Mtree(BaseIndex):

    def __init__(self, PATH="./MTree1M-int2000", labels=["L1", "L2"], descriptor_values=282, knn_gts="knn_gt.json"):
        super().__init__(PATH, labels, descriptor_values, knn_gts)


    def get_distance(self, priority_queue, L1_regions, query_row, df_orig, labels, is_profi=True):
        for (pivot_id, ip, radius) in np.array(L1_regions):
            pivot_row = df_orig[df_orig["object_id"] == pivot_id]
            if is_profi:
                priority_queue.append([str(ip), self.get_euclidean_distance(query_row.drop(labels, axis=1), pivot_row.drop(labels, axis=1)) - np.float(radius)])
            else:
                priority_queue.append([str(ip), get_cophir_distance(query_row.drop(labels, axis=1).values[0].T, pivot_row.drop(labels, axis=1).values[0].T) - np.float(radius)])
        priority_queue = sorted(priority_queue, key=lambda x: x[1])
        return priority_queue