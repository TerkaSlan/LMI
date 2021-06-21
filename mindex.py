from BaseIndex import BaseIndex
from imports import np
from cophir_distances import get_cophir_distance

class Mindex(BaseIndex):

    def __init__(self, PATH="./MTree1M-int2000", labels=["L1", "L2"], descriptor_values=282, knn_gts="knn_gt.json"):
        super().__init__(PATH, labels, descriptor_values, knn_gts)
        

    def get_distance(self, pivot_ids, pivot_descriptors, query_row, df_orig, labels, prev_pivot=None, max_levels=2, is_profi=True):
        ## TBD: make common with Mtree
        pivot_distances = []
        max_pivot_ip = 127
        for pivot_ip, (pivot_id, pivot_descriptor) in enumerate(zip(pivot_ids, pivot_descriptors)):
            if is_profi:
                pivot_distances.append([str(pivot_ip), self.get_euclidean_distance(query_row.drop(labels, axis=1), pivot_descriptor.reshape(1, -1))])
            else:
                pivot_distances.append([str(pivot_ip), get_cophir_distance(query_row.drop(labels, axis=1).values[0].T, pivot_descriptor.T)])
            if pivot_ip == max_pivot_ip:
                break
        pivot_distances = sorted(pivot_distances, key=lambda x: x[1])
        #for k in sorted(d, key=d.get, reverse=False):
        #    k, d[k]
        #pivot_distances = OrderedDict(sorted(pivot_distances.items(), key = itemgetter(1), reverse = False))
        #print(type(pivot_distances))
        return pivot_distances


    def get_wspd(self, priority_queue, pivot_ids, pivot_distances, df_orig, labels, L1_only_pivots, existing_regions, pow_list, prev_pivot, max_levels=2, is_profi=True):
        p_area = prev_pivot[1]#pivot_distances[prev_pivot]
        level_path = [int(x) for x in prev_pivot[0].split(".")]
        level = len(level_path)
        print(level_path, sum(pow_list[:level+1]))

        for p in pivot_distances:
            if (p[0] != prev_pivot[0]) and f"{prev_pivot[0]}.{p[0]}" in existing_regions_dict:
                #(prev_pivot[1] + p[1]*pow_list[level]) / 1.75
                priority_queue.append([prev_pivot[0] + "." + p[0], (p_area + p[1]*pow_list[level]) / sum(pow_list[:level+1])])
        priority_queue = sorted(priority_queue, key=lambda x: x[1])
        return priority_queue