import numpy as np

def train_val_test_split(dataframe):
    train_length, val_length = int(dataframe.shape[0] * 0.65), int(dataframe.shape[0] * 0.15)

    train_dataframe = dataframe.iloc[:train_length, :]
    val_dataframe = dataframe.iloc[train_length:train_length + val_length, :]
    test_dataframe = dataframe.iloc[train_length + val_length:, :]

    return (train_dataframe, val_dataframe, test_dataframe)

def get_unique_categorical_counts(dataset, cont_count=10):
    return dataset.iloc[:, 0:-(cont_count + 1)].nunique().to_list()

def get_target_values(dataframe, positive_class=1):
    targets_source = dataframe.iloc[:, -1:]
    target = np.zeros(targets_source.shape)
    for i in range(targets_source.shape[0]):
        target[i] = 1 if targets_source.iloc[i, 0] == positive_class else 0

    return target

def get_cont_values(dataframe, cont_count=10):
    cont = dataframe.iloc[:, -(cont_count + 1):-1]
    return cont.to_numpy()

def get_categ_values(dataframe, cont_count=10):
    categ = dataframe.iloc[:, 0:-(cont_count + 1)]

    for i in range(categ.shape[1]):
        categ[categ.columns[i]] = categ[categ.columns[i]].astype("category").cat.codes

    return categ.to_numpy()

def get_categ_cont_target_values(dataframe, positive_class=1, cont_count=10):
    target = get_target_values(dataframe, positive_class)
    cont = get_cont_values(dataframe, cont_count)
    categ = get_categ_values(dataframe, cont_count)

    return (cont, categ, target)
