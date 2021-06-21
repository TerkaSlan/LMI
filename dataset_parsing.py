from imports import pd, re, np
from sklearn import preprocessing

BASE_DATASET_DIR="/storage/brno6/home/tslaninakova/learned-indexes/"

def load_indexes(filename, names):
    return pd.read_csv(filename, names = names + ["object_id"],  sep=r'[.\s+]', engine='python', header=None)

def get_num_of_numerical_rows(line, delimiter=";", col_to_skip="messif.objects.impl.ObjectGPSCoordinate"):
    """Determines the number of numerical rows that'll follow based on the number
    of columns in a current line. If there's a `col_to_skip`, returns position
    of its associated data
    """
    column_names = line.split(delimiter)
    num_of_rows = len(column_names)//2
    row_to_skip = None
    if col_to_skip in column_names:
        row_to_skip = column_names.index(col_to_skip) // 2
    return num_of_rows, row_to_skip

def get_label(h_line):
    """Gets the label (`object_id`) from the first column by searching a phrase
    just before it.
    """
    re_match = re.search('BucketIdObjectKey ', h_line)
    if re_match:
        return int(h_line[re_match.end():])
    else:
        re_match = re.search('AbstractObjectKey ', h_line)
        if re_match:
            return int(h_line[re_match.end():])
        else: return None

def parse_objects(filename="objects.txt"):
    """Loads a file with objects line by line, extracting labels (object_id) and
    numerical data. Returns list of labels and list of numerical values, merged to
    a single row (so the information about which value corresponds to which column
    is lost).
    """
    labels = []; numerical = []; numerical_row = []; attributes_per_descr_len = []
    counter = 0
    next_line = 2
    with open(filename) as file:
        line = file.readline().rstrip('\n')
        while line:
            # the 0th line contains the label
            if counter == 0:
                labels.append(get_label(line))
                counter += 1
            # the 2nd line contains column names = how many numerical rows will follow
            elif counter == next_line:
                num_of_rows, row_to_skip = get_num_of_numerical_rows(line) 
                # get a list of integers of all the consecutive numerical rows
                for n in range(num_of_rows):
                    line = file.readline().rstrip('\n')
                    #n_of_descriptors = num_of_rows - 1 if row_to_skip is not None else num_of_rows
                    if not row_to_skip or n != row_to_skip:
                        found_attributes = list(map(int, re.findall(r'[\-0-9\.]+', line)))
                        if len(attributes_per_descr_len) < num_of_rows - int(row_to_skip is not None):
                            attributes_per_descr_len.append(len(found_attributes))
                        numerical_row += found_attributes
                counter = 0
                numerical.append(numerical_row)
                numerical_row = []
            
            else:
                counter += 1
            line = file.readline().rstrip('\n')
        return labels, numerical, attributes_per_descr_len

def merge_dfs(normalized, labels, index_df):
    """Merges normalized numerical data, labels and dataframe of indexes to one
    dataframe. All the columns from both of them are kept.
    """
    df = pd.DataFrame(normalized)
    df['object_id'] = labels
    # Move "object_id" column to the front
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    final = pd.merge(df, index_df, on=['object_id'], how = 'outer')[:len(normalized)]
    # move first-level and second-level columns to the front
    cols = final.columns.tolist()
    final = final[cols[-2:] + cols[:-2]]
    return final

def get_objects_with_indexes(names, indexes_filename='level-2.txt', objects_filename='objects.txt'):
    """ Gets values of descriptors and indexes (labels), combines them into a pandas df.
    """
    index_df = load_indexes(indexes_filename, names)
    labels, numerical, descr_lengths = parse_objects(objects_filename)
    df = merge_dfs(numerical, labels, index_df)
    return df, descr_lengths

def scale_per_descriptor(df, labels, descriptor_value_counts):
    """
    Scales the descriptor values per descriptors using sklearn.preprocessing.scale 
     - centers to the mean and ensures unit variance.
    'Per descriptor' means that there are 5 descriptors of different lengths in CoPhIR dataset.
    The scaling is done individually per all of these 5.

    Parameters
    ----------
    df : DataFrame
        Dataset.
    descriptor_value_counts : list
        Number of individual values in each descriptor.
        
    Returns
    -------
    DataFrame
        Normalized dataset.
    """
    col_pos = 0
    normalized = []
    numerical = df.drop(labels+["object_id"], axis=1).values
    for descriptor_value in descriptor_value_counts:
        current = numerical[:, col_pos:col_pos+descriptor_value]
        normalized.append(preprocessing.scale(current))
        col_pos += descriptor_value
    df = df.drop(df.columns.difference(labels+["object_id"]), 1)
    df = pd.concat([df, pd.DataFrame(np.hstack((normalized)))], axis=1)
    return df

def load_indexes_profiset(base_dir, filenames=['level-1.txt', 'level-2.txt'], is_mocap=False):
    """ Loads a DataFrame with labels and object_ids.
    
    Starts with loading .csv files from smallest and merging with   

    Parameters
    ----------
    base_dir : String
        Directory of where to look for level-*.txt datasets
    filenames : list
        List of filenames
    
    Returns
    -------
    DataFrame
        Normalized dataset.
    """
    label_names = [f"L{i}" for i in range(1, len(filenames)+1)]
    
    df_i = pd.DataFrame([])
    filenames.reverse()
    for c, filename in enumerate(filenames):
        if c != 0: col_names = label_names[:-c]
        else: col_names = label_names
        if is_mocap:
            df_ = pd.read_csv(base_dir+filename, names = col_names + ["object_id"], sep=r'[.+\s]', engine='python', header=None)
        else:
            df_ = pd.read_csv(base_dir+filename, names = col_names + ["object_id"], sep=r'[.+\s]', engine='python', dtype=np.int64, header=None)
        df_i = pd.concat([df_i, df_])
        df_i = df_i.drop_duplicates(["object_id"])
    return df_i.apply(pd.to_numeric, errors='ignore')

def get_profiset(objects_path="datasets/descriptors-decaf-odd-5M-1.data", 
                 indexes_path="MtreeProfi2000/"):

    objects_path=f"{BASE_DATASET_DIR}/{objects_path}"
    indexes_path=f"{BASE_DATASET_DIR}/{indexes_path}"
    print("Loading labels")
    index_df = load_indexes_profiset(indexes_path, filenames=[f'level-{l}.txt' for l in range(1,3)])
    index_df = index_df.sort_values(by=["object_id"])
    assert index_df.shape[0] == 1000000
    print("Loading descriptors")
    data = pd.read_csv(objects_path, header=None, sep=" ", dtype=np.float16)
    data = data.drop(data.columns[-1], axis=1)
    data.reset_index(drop=True, inplace=True)
    index_df.reset_index(drop=True, inplace=True)
    df_full = pd.concat([data, index_df], axis=1)
    df_full = df_full.sample(frac=1)
    return df_full

def get_mocap(objects_path="MTree1M-mocap/"):
    objects_path=f"{BASE_DATASET_DIR}/{objects_path}"
    print("Loading labels")
    index_df = load_indexes_profiset(objects_path, filenames=[f'level-{l}.txt' for l in range(1,3)], is_mocap=True)
    index_df["L2"] = index_df["L2"].astype(np.int64)
    index_df["L1"] = index_df["L1"].astype(np.int64)
    index_df = index_df.sort_values(by=["object_id"])
    print("Loading descriptors")
    df = pd.read_csv(f"{objects_path}objects-even.txt", header=None)
    object_ids = pd.read_csv(f"{objects_path}objects-odd.txt", sep=r'[.+\s]', engine='python', header=None)
    df["object_id"] = object_ids[5]
    df = index_df.merge(df, how='left',on=['object_id'])
    df = df[~df.duplicated(subset=['object_id'])]
    df = df.sample(frac=1)
    assert df.shape[0] == 354902
    return df