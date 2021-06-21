from imports import np, pd, json, time, logging, warnings, ConvergenceWarning
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

from dataset_parsing import get_objects_with_indexes, scale_per_descriptor, get_profiset, get_mocap
from utils import get_knn_objects, one_hot_frequency_encode, label_encode_data, label_encode_vectors

# Training and prediction
from RandomForest import LMIRandomForest
from LogisticRegression import LMILogisticRegression
from NeuralNetwork import LMINeuralNetwork
from BaseIndex import BaseIndex
from GMM import GaussianMixtureModel

logging.basicConfig(datefmt='%d-%m-%y %H:%M', format='%(asctime)-15s%(levelname)s: %(message)s', level=logging.INFO)

class LMI(BaseIndex):
    """
    Base class representing Learned metric index and basic operation on it - training and searching

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
        super().__init__(PATH, labels, knn_gts)
        """
        Parameters
        ----------
        [TBD]
    
        """
        self.stack = None
        self.mapping = None
        self.classifier = None
        self.encoders = []
        self.objects_in_buckets = {}
        self.predict_drop = self.labels + [f"{l}_pred" for l in self.labels] + ["object_id"]
        
    def get_dataset(self, dataset_path=None, normalize=True, return_original=False):
        """
        1. Loads the dataset, assumes objects and labels are stored by their standard names in `PATH`
        or that `dataset_path` is provided
        
        2. Normalizes the descriptior value with z-score normalization, 
            read more: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html
        
        Parameters
        ----------
        dataset_path : str
            Path to the dataset location. If None, the dataset in self.path is used.
        normalize : boolean
            Decides whether to perform z-score normalization, True by default
        return_original : boolean
            Decides whether to return the non-normalized dataset as well, False by default

        Returns
        -------
        DataFrame
            Pandas DF of merged descriptors (normalized) and labels.
        """
        if dataset_path is None:
            dataset_path = self.dir
        
        if "mocap" in dataset_path.lower():
            print("Loading Mocap dataset.")
            df = get_mocap()
            df_orig = df
        elif "profi" in dataset_path.lower():
            print("Loading Profiset dataset.")
            df = get_profiset()
            df_orig = df
        else:
            print("Loading CoPhIR dataset.")
            df_orig, attr_lengths = get_objects_with_indexes(self.labels, f'{dataset_path}/level-{str(self.n_levels)}.txt', f'{dataset_path}/objects.txt')
            if normalize:
                df = scale_per_descriptor(df_orig, self.labels, attr_lengths)
            else:
                df = df_orig
        
        assert df.shape[1] == self.descriptor_values + self.n_levels + len(["object_id"])
        logging.info(f"Loaded dataset of shape: {df.shape}")
        if return_original:
            return df, df_orig
        else:
            return df

    def get_knn_ground_truth(self, filename=None):
        """ Loads object 30-NN ground truth file (json).

        Returns
        -------
        gt_knns: dict
            Ground truth for 30 kNN query
        """
        if not filename:
            filename = self.knn_gts_file
        with open(filename) as json_file:
            gt_knns = json.load(json_file)
        return gt_knns
    
    def get_sample_1k_objects(self, df, path="./datasets/queries.data"):
        """ Gets portion of the dataset with predictions according to random sample of 1000 objects. Will be used for knn search run.

        Parameters
        -------
        df: Pandas DataFrame
            Dataset with predictions
        path: String
            Path to the queries file
        Returns
        -------
        Pandas DataFrame
            Subset of the original dataset with 1000 queries
        """
        return df[df["object_id"].isin(get_knn_objects(path=path))]
    
    def prepare_data_for_training(self, model_dict, df, level=0):
        """ Splits the DataFrame into training values (X) and labels (y)
        
        If the desired model is multilabel NN, target labels are composed of 30-NN as opposed to 1-NN.

        Parameters
        -------
        model_dict: Dict
            Dictionary of models specification
        df: Pandas DataFrame
            Dataset
        
        Returns
        -------
        X: Numpy array of shape (n_objects x n_descriptor_values)
            Training values
        y: Numpy array of shape (n_objects x 1) or (n_objects x 30) for multilabel
            Training labels 
        
        """
        if "nn-multi" in model_dict:
            labels = [f"L{i}_labels" for i in range(1, len(self.labels)+1)]
            X = df.drop(self.labels + [f"{l}_pred" for l in self.labels] + ["object_id"] + labels, axis=1, errors="ignore").values
            y = one_hot_frequency_encode(df[labels[0]].values, n_cats=df[self.labels[0]].max())
        else:
            X = df.drop(self.labels + [f"{l}_pred" for l in self.labels] + ["object_id"], axis=1, errors="ignore").values
            y = df[self.labels[level]].values
        
        return X,y

    def train(self, df, model_dict, pretrained_root=False, should_shuffle_dataset=True, should_erase=False, na_label=None):
        """ Train the whole LMI.
        1. Prepares the data
        2. Chooses the model to use for training
        
        If the desired model is multilabel NN, target labels are composed of 30-NN as opposed to 1-NN.

        Parameters
        -------
        model_dict: Dict
            Dictionary of models specification
        df: Pandas DataFrame
            Dataset
        
        Returns
        -------
        X: Numpy array of shape (n_objects x n_descriptor_values)
            Training values
        y: Numpy array of shape (n_objects x 1) or (n_objects x 30) for multilabel
            Training labels 
        
        """
        if self.stack and not should_erase:
            logging.warn(f"self.stack wasn't empty - training would erase already trained models. If you want to erase them anyway, set should_erase=True")
            return
        
        self.stack = []; self.mapping = []
        if should_shuffle_dataset:
            df = df.sample(frac=1)
        
        X, y = self.prepare_data_for_training(model_dict, df)
        y_obj_id = df["object_id"].values

        if "RF" in model_dict:
            self.classifier = LMIRandomForest()
            model, predictions, encoder = self.classifier.train(model_dict["RF"][0], X, y, self.descriptor_values)
        elif "LogReg" in model_dict:
            self.classifier = LMILogisticRegression()
            model, predictions, encoder = self.classifier.train(model_dict["LogReg"][0], X, y, self.descriptor_values)
        elif "NN" in model_dict:
            self.classifier = LMINeuralNetwork()
            model, predictions, encoder = self.classifier.train(model_dict["NN"][0], X, y, self.descriptor_values)
        elif "NNMult" in model_dict:
            self.classifier = LMINeuralNetwork()
            model, predictions, encoder = self.classifier.train(model_dict["NNMult"][0], X, y, self.descriptor_values, \
                                                        loss='categorical_crossentropy', metric='categorical_accuracy')
        elif "GMM" in model_dict:
            self.classifier = GaussianMixtureModel()
            model, predictions, encoder = self.classifier.train(model_dict["GMM"][0], X, y, self.descriptor_values)
        else:
            logging.warn(f"Did not recognize any known classifier from {model_dict}, exiting.")
            return None

        self.encoders.append(encoder)
        df_l1 = pd.DataFrame(np.array([predictions, y_obj_id]).T, columns=[f"{self.labels[0]}_pred", "object_id"])
        df_res = df.merge(df_l1, on="object_id")
        self.stack.append(model)
        
        for i, label in enumerate(self.labels[1:]):
            group_cond = [f"{self.labels[j]}_pred" for j in range(i+1)]
            groups = df_res[(~df_res[label].isna()) & (df_res[label] != na_label)].groupby(group_cond)
            stack_l1, df_l1, mapping_l1 = self.train_level(df_res, groups, model_dict, level=i+1)
            df_res = df_res.merge(df_l1[["object_id", f"{label}_pred"]], on=["object_id"], how="left")
            self.mapping.append(mapping_l1)
            self.stack.append(stack_l1)

        self.objects_in_buckets = self.get_buckets(df_res)

        assert len(self.stack) == len(self.labels)
        cols = df_res.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        return df_res[cols]
    
    def train_level(self, df, groupby_df, model_dict, level=1):
        """ Train one LMI level
        1. Iterates the groups of the dataset. These groups are created based on groupby of the predictions on the
        2. Chooses the model to use for training
        
        If the desired model is multilabel NN, target labels are composed of 30-NN as opposed to 1-NN.

        Parameters
        -------
        model_dict: Dict
            Dictionary of models specification
        df: Pandas DataFrame
            Dataset
        
        Returns
        -------
        X: Numpy array of shape (n_objects x n_descriptor_values)
            Training values
        y: Numpy array of shape (n_objects x 1) or (n_objects x 30) for multilabel
            Training labels 
        
        """
        stack_l = []; preds_l = []; object_ids = []; mapping = []; self.encoders.append([])
        y_label = self.labels[level]
        logging.info(f"Training level {level}")
        for name, group in groupby_df:
            object_ids.extend(group["object_id"].values)
            X, y = self.prepare_data_for_training(model_dict, group, level=level)
            assert X.shape[1] == 282 or X.shape[1] == 4096
            
            if "RF" in model_dict:
                model, predictions, encoder = self.classifier.train(model_dict["RF"][level], X, y, self.descriptor_values)
            elif "LogReg" in model_dict:
                model, predictions, encoder = self.classifier.train(model_dict["LogReg"][level], X, y, self.descriptor_values)
            elif "NN" in model_dict:
                model, predictions, encoder = self.classifier.train(model_dict["NN"][level], X, y, self.descriptor_values,\
                                                                    input_data_shape=X.shape[1], output_data_shape=max(y)+1)
            elif "NNMult" in model_dict:
                model, predictions, encoder = self.classifier.train(model_dict["NNMult"][level], X, y, self.descriptor_values, \
                                                                    loss='categorical_crossentropy', metric='categorical_accuracy')
            elif "GMM" in model_dict:
                model, predictions, encoder = self.classifier.train(model_dict["GMM"][level], X, y, self.descriptor_values)
            self.encoders[-1].append(encoder)
            
            if type(name) is float or type(name) is int:
                mapping.append(tuple([int(name)]))
            else:
                mapping.append(tuple([int(n) for n in name]))
            stack_l.append(model)
            preds_l.extend(predictions)
            
        df_l = pd.DataFrame(np.array([preds_l, object_ids]).T, columns=[y_label+"_pred"] + ["object_id"])
        df_l = df.merge(df_l, on="object_id")
        return stack_l, df_l, mapping
    
    def add_to_priority_queue(self, priority_q, predictions, level=1, parent_nodes=[]):
        """ Adds collected predictions to priority queue.

        Parameters
        -------
        priority_q: List
            Priority queue
        predictions: numpy array
            collected predictions
        level: int
            Level of the nodes
        
        Returns
        -------
        Modified priority_q
        """
        start = "M" if level != len(self.labels) else "C"
        parent_string = ""
        for node in parent_nodes:
            parent_string += node[len("M.1."):] + "."
        for prediction in predictions:
            priority_q.append({f"{start}.1.{parent_string}" + str(prediction['value_l']): prediction['votes_perc']})
        
        return priority_q
        
    def collect_probs_for_node(self, node_label, query):
        """ Collects probabilities for a given model.

        Parameters
        -------
        node_label: str
            label of the node searched
        query: Int
            searched query
        
        Returns
        -------
        predictions
        """
        level = len(node_label) - 2
        predictions = []

        path_label = tuple([int(node_label[-i]) for i in range(1, level+1)])
            
        if path_label in self.mapping[level-1]:
            stack_index = self.mapping[level-1].index(path_label)
            model = self.stack[level][stack_index]
            predictions = self.classifier.predict(query, model, encoder=self.encoders[1][stack_index])
        else:
            logging.warn(f"{path_label} is not in self.mapping[{[level-1]}].")
        return predictions

    def process_node(self, priority_q, query, debug=False):
        """ Gets top node from priority queue, collects its children, adds them to priority queue.

        Parameters
        -------
        priority_q: List
            Priority queue
        query: Int
            searched query
        
        Returns
        -------
        priority_q
        node - popped node
        """
        popped = priority_q.pop(0)
        node = list(popped.keys())[0]
        node_label = node.split('.')        
        if node[0] == "C": 
            if debug:
                logging.info(f"L{len(node_label) - 2} found bucket {node}")  
            return priority_q, node
        
        if debug:
            logging.info(f"Popped {node}")

        predictions = self.collect_probs_for_node(node_label, query)
        priority_q = self.add_to_priority_queue(priority_q, predictions, len(node_label) - 1, [node])
        priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
        if debug:
            logging.info(f"L{len(node_label) - 1} added - PQ (Top 5): {priority_q[:5]}\n")    
    
        return priority_q, node
    
    def search(self, df_res, query_id, stop_cond_objects=None, debug=False):
        """ Runs the searching procedure.
        
        For a given query (object from the trained dataset), searches the LMI space
        and return the visited nodes.

        Parameters
        -------
        df_res: DataFrame
            DataFrame of the trained LMI
        query_id: Int
            ID of the query searched
        stop_cond_objects: List
            List of stop conditions: number of objects in the visited buckets, 
                                     when the searching procedure should create a checkpoint.
        Returns
        -------
        Dict of:
        - `id` for node id (= `object_id`)
        - `time_checkpoints` time (in s) it took to find the corresponding checkpoints
        - `popped_nodes_checkpoints` - the nodes that managed to be popped till their 
                                       collective sum of objects did not overstep the corresponding 
                                       `stop_cond_objects` threshold
        - `objects_checkpoints` - the actual sum of all found objects following `stop_cond_objects`. 
                                  Is slightly higher than `stop_cond_objects`
        """
        s = time.time()
        time_checkpoints = []; popped_nodes_checkpoints = []; objects_checkpoints = []
        query_row = df_res[(df_res['object_id'] == query_id)]
        query = query_row.drop(self.predict_drop, axis=1).values
        predictions = self.classifier.predict(query, self.stack[0], self.encoders[0])
        priority_q = []
        priority_q = self.add_to_priority_queue(priority_q, predictions)
        if debug: logging.info(f"Step 1: L1 added - PQ: {priority_q}\n")

        current_stop_cond_idx = 0

        popped_nodes = []
        iterations = 0; n_steps = 0
        while len(priority_q) != 0:
            if stop_cond_objects != None and len(stop_cond_objects) == current_stop_cond_idx:
                return {'id': int(query_id), 'time_checkpoints': time_checkpoints, 'popped_nodes_checkpoints': popped_nodes_checkpoints, 'objects_checkpoints': objects_checkpoints}
            else:
                priority_q, popped = self.process_node(priority_q, query, debug=debug)
                if type(popped) is list:
                    popped_nodes.extend(popped)
                else: popped_nodes.append(popped)

                if stop_cond_objects is not None:
                    index = tuple([int(p) for p in popped.split('.')[2:]])
                    if len(index) == 1: index = index[0]
                    if index in self.objects_in_buckets:
                        n_steps += self.objects_in_buckets[index]
                    else:
                        n_steps += 0
                    if current_stop_cond_idx < len(stop_cond_objects) and stop_cond_objects[current_stop_cond_idx] <= n_steps:
                        time_checkpoint = time.time()
                        time_checkpoints.append(time_checkpoint-s)
                        popped_nodes_checkpoints.append(popped_nodes.copy())
                        objects_checkpoints.append(n_steps)
                        current_stop_cond_idx += 1
            iterations += 1
        return {'id': int(query_id), 'steps': n_steps, 'time_checkpoints': time_checkpoints, 'popped_nodes_checkpoints': popped_nodes_checkpoints, 'steps_checkpoints': objects_checkpoints}