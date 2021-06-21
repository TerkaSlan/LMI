from utils import label_encode_data
from random import shuffle
from imports import np, json, re

class BaseIndex(object):
    """
    Base class representing Index and basic operation on it - training and searching

    ...

    Attributes [TBD]
    ----------
    PATH : str
        Path to the directory with data, expects directory to be descriptive of dataset/index type 
        (e.g. to contain 'mindex' and/or 'profiset' in name if it's mindex and/or profiset)
            - expects to find `objects.txt` file in the dir and file with labels: `level-x.txt` 
    labels : str
        Names for labels to use, also denotes how many labels (levels of the tree) are there
    knn_gts : str
        Filename of k-nearest neighbors file

    Methods
    -------
    get_dataset(dataset_path=None)


    """
    def __init__(self, PATH="./Mtree", labels=["L1", "L2"], knn_gts="knn_gt.json"):
        """
        Parameters
        ----------
        [TBD]
    
        """
        self.dir = PATH
        self.n_levels = len(labels)
        self.labels = labels
        self.knn_gts_file = f"{self.dir}/{knn_gts}"
        self.is_profiset_or_mocap = True if ("profi" in PATH.lower() or "mocap" in PATH.lower()) else False
        self.is_mindex = True if "mindex" in PATH.lower() else False
        self.descriptor_values = 282 if not self.is_profiset_or_mocap else 4096
        
    def search(self, prob_distr, encoder, value="value_l"):
        pass
    
    def get_buckets(self, df_res):

        if self.is_mindex:
            groupbys = []
            current_df = df_res.copy()
            for i, label in enumerate(self.labels):
                if len(self.labels) == i+1:
                    L_na = current_df[~current_df[f"{self.labels[i+1]}_pred"].isna()].groupby([f"{l}_pred" for l in self.labels])
                else:
                    current_df = current_df[current_df[f"{self.labels[i+1]}_pred"].isna()]
                    L_na = current_df.groupby([f"{label}_pred"])
                groupbys.append(L_na)
            
        else:
            group_cond = [f"{self.labels[i]}_pred" for i in range(len(self.labels))]
            groupbys = df_res.groupby(group_cond).size()
        
        bucket_dict = {}
        for key, val in zip(groupbys.keys().values, groupbys.values):
            bucket_dict[key] = val 
        
        return bucket_dict
    
    def get_euclidean_distance(self, object_1, object_2):
        assert object_1.shape == object_1.shape and object_1.shape[1] == 4096
        return np.linalg.norm(object_1-object_2)
    
    def load_knns(self, path=None):
        """ Loads object 30 knns ground truth file (json).

        Parameters
        ---------
        df: Pandas DataFrame
            Loaded dataset
        path: String
            Path to where the 1000 queries are stored.

        Returns
        -------
        df: Pandas DataFrame
            Subset of the original dataset containing just the 1000 queries.
        
        """
        if path is None:
            path = self.knn_gts_file
        with open(path) as json_file:
            gt_knns = json.load(json_file)
        return gt_knns
    
    def get_queries(self, df, path="/storage/brno6/home/tslaninakova/learned-indexes/datasets/queries.data", should_be_int=True):
        """ Loads 1000 random objects, joins them with an existing DataFrame.

        Returns
        -------
        df: Pandas DataFrame
            Loaded dataset
        path: String
            Path to where the 1000 queries are stored.
        """
        knn_object_ids = []
        with open(path) as f:
            for line in f.readlines():
                z_1 = re.findall(r"AbstractObjectKey ([\d_-]+)", line)
                if z_1:
                    if should_be_int:
                        knn_object_ids.append(int(z_1[0]))
                    else:
                        knn_object_ids.append(z_1[0])
        if should_be_int:
            return df[df["object_id"].isin(np.array(knn_object_ids, dtype=np.int64))]
        else:
            return df[df["object_id"].isin(np.array(knn_object_ids))]