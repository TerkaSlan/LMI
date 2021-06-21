from imports import np

def guess_the_labels(df_res):
    """ Infers the labels to be used in knn evaluation from the dataset
    - if there are '_pred' columns, those are the labels
    - else: the "Lx" labels are the target labels (in case of original Mtree/Mindex)

    Parameters
    -------
    df_res: DataFrame
        Trained LMI DataFrame
    
    Returns
    -------
    labels: List
        List of infered labels
    """
    labels = []
    labels = [col for col in df_res.columns if type(col) is str and col[-5:] == "_pred"]
    if labels == []:
        i = 1
        while f"L{i}" in df_res.columns:
            labels.append(f"L{i}")
            i += 1
    return labels

def knn_evaluate(popped, query_id, df_res, knns, debug=False):
    """ Goes over the overlap of found buckets and target buckets of the ground-truth knns,
        counts their objects and evaluates the actual knn preformance.

    Parameters
    -------
    popped: List
        List of popped nodes
    query_id: Int
        ID of the query searched
    df_res: DataFrame
        Trained LMI DataFrame
    knns: Dict
        Buckets of the query's knns
    
    Returns
    -------
    labels: List
        List of infered labels
    """
    knns_found = 0
    for p in popped:
        if p in knns:
            print(p)
            knns_found += len(knns[p])
    if debug:
        print(f"N. of knns found: {knns_found} in {len(popped)} buckets.")
    return knns_found / 30

def get_knn_buckets_for_query(df_res, query, gt_knn, labels=None):
    """ Goes over the ground-truth knns and finds their respetive buckets,
        collects them into a dictionary.

    Parameters
    -------
    query: Int
        ID of the query searched
    df_res: DataFrame
        Trained LMI DataFrame
    gt_knn: Dict
        Query's knns
    labels: List
        List of bucket labels
    
    Returns
    -------
    d: Dict
        Dict of knn's buckets.
    """
    if not labels:
        labels = guess_the_labels(df_res)
    knns = gt_knn[str(query)]
    d = {}
    for k in knns.keys():
        bucket = "C.1." + "".join([f"{int(k)}." for k in df_res[df_res["object_id"] == int(k)][labels].values[0] if str(k) != "nan"])[:-1]
        if bucket in d:
            d[bucket].append(k)
        else:
            d[bucket] = [k]
    return d

def evaluate_knn_per_query(res, df_res, gt_knns, labels=None, debug=True):
    """ Finds and evaluates knn performance of a single query on multiple checkpoints.

    Parameters
    -------
    res: Dict
        result of the searching procedure
    df_res: DataFrame
        Trained LMI DataFrame
    gt_knns: Dict
        All the queries' knns
    labels: List
        List of bucket labels
    
    Returns
    -------
    knn_results_per_query: List
        List of knn recalls.
    """
    if debug:
        print(f"Evaluating k-NN performance on {len(res['objects_checkpoints'])} checkpoints: {res['objects_checkpoints']}")
    if not labels:
        labels = guess_the_labels(df_res)
    query = res['id']
    knns = get_knn_buckets_for_query(df_res, query, gt_knns, labels)

    knn_results_per_query = []
    for i, checkpoint in enumerate(res['objects_checkpoints']):
        knn_results_per_query.append(knn_evaluate(res["popped_nodes_checkpoints"][i], query, df_res, knns, debug=debug))
    return knn_results_per_query

def evaluate_knn(gt_knns, results, df_res, labels=None, object_labels=[500,1000,3000,5000,10000,50000,100000,200000,300000,500000]):
    """ Finds and evaluates knn performance of multiple queries on multiple checkpoints.

    Parameters
    -------
    results: Dict
        results of the searching procedure
    df_res: DataFrame
        Trained LMI DataFrame
    gt_knns: Dict
        All the queries' knns
    labels: List
        List of bucket labels
    
    Returns
    -------
    knn_results: List
        List of knn recalls.
    time_results: List
        List of times.
    steps_results: List
        List of steps.
    """
    knn_results = []
    if not labels:
        labels = guess_the_labels(df_res)

    for res in enumerate(results):
        knn_results_per_query = evaluate_knn_per_query(res, df_res, gt_knns, labels)
        knn_results.append(knn_results_per_query)

    time_results =  [[0] + result['time_checkpoints'] for result in results]
    steps_results =  [[0] + result['steps_checkpoints'] for result in results]

    knn_results = np.array(knn_results); time_results = np.array(time_results)
    print([knn_results[:, i].mean() for i in range(knn_results.shape[1])])

    return knn_results, time_results, steps_results