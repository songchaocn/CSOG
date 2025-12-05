import numpy as np

import torch
import os
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity


def get_data(folder, correlation, name, avg_degree=10):

    folderpath = './data/' + folder + '/'
    y_train = np.loadtxt(os.path.join(folderpath, "labels_tr.csv"), delimiter=",").astype(int)
    y_test = np.loadtxt(os.path.join(folderpath, "labels_te.csv"), delimiter=",").astype(int)
    y = np.concatenate((y_train, y_test))
    x_train_set = []
    x_test_set = []

    x_train = np.loadtxt(os.path.join(folderpath, f"{name}_tr.csv"), delimiter=",")
    x_test = np.loadtxt(os.path.join(folderpath, f"{name}_te.csv"), delimiter=",")
    x = np.concatenate((x_train, x_test))
    if correlation == 'spearman':
        corr,_ = stats.spearmanr(x)
    elif correlation == 'similarity':
        corr = cosine_similarity(x.T)



    np.fill_diagonal(corr, -np.inf)

    rows, cols = np.triu_indices_from(corr, k=1)
    values = corr[rows, cols]

    sorted_indices = np.argsort(values)[::-1]
    rows, cols, values = rows[sorted_indices], cols[sorted_indices], values[sorted_indices]

    num_nodes = x.shape[1]
    target_edges = avg_degree * num_nodes // 2

    selected_rows = rows[:target_edges]
    selected_cols = cols[:target_edges]
    selected_values = values[:target_edges]

    edge_index = np.vstack([np.concatenate([selected_rows, selected_cols]),
                            np.concatenate([selected_cols, selected_rows])])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_weights = np.concatenate([selected_values, selected_values])
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    x_train, x_test, y_train, y_test = train_test_split(x, y.T, test_size=0.3, shuffle=True)

    for i in range(x_train.shape[0]):
        x_train_set.append(Data(x=torch.tensor(x_train[i, :], dtype=torch.float).unsqueeze(1),
                                y=torch.tensor(y_train[i], dtype=torch.long), edge_index=edge_index,
                                edge_weight=edge_weight).to('cuda'))
    for i in range(x_test.shape[0]):
        x_test_set.append(Data(x=torch.tensor(x_test[i, :], dtype=torch.float).unsqueeze(1),
                               y=torch.tensor(y_test[i], dtype=torch.long), edge_index=edge_index,
                               edge_weight=edge_weight).to('cuda'))

    train_data_loader = DataLoader(x_train_set, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(x_test_set, batch_size=1, shuffle=False)

    return train_data_loader, test_data_loader



def get_sample_graph(folder, name, avg_degree=10):
    # 构建sample to sample graph
    folderpath = './data/' + folder + '/'
    x_train = np.loadtxt(os.path.join(folderpath, f"{name}_tr.csv"), delimiter=",")
    x_test = np.loadtxt(os.path.join(folderpath, f"{name}_te.csv"), delimiter=",")
    if name in (1, 2, 3):
        y_train = np.loadtxt(os.path.join(folderpath, "labels_tr.csv"), delimiter=",").astype(int)
        y_test = np.loadtxt(os.path.join(folderpath, "labels_te.csv"), delimiter=",").astype(int)
        x_train = MinMaxScaler().fit_transform(x_train)
        x_test = MinMaxScaler().fit_transform(x_test)
    else:
        y_train = np.loadtxt(os.path.join(folderpath, "4_tr_label.csv"), delimiter=",").astype(int)
        y_test = np.loadtxt(os.path.join(folderpath, "4_te_label.csv"), delimiter=",").astype(int)

        x_train = MinMaxScaler().fit_transform(x_train)
        x_test = MinMaxScaler().fit_transform(x_test)
    y = np.concatenate((y_train, y_test))



    x = np.concatenate((x_train, x_test))
    corr = cosine_similarity(x)


    np.fill_diagonal(corr, -np.inf)

    rows, cols = np.triu_indices_from(corr, k=1)
    values = corr[rows, cols]

    sorted_indices = np.argsort(values)[::-1]
    rows, cols, values = rows[sorted_indices], cols[sorted_indices], values[sorted_indices]

    num_nodes = x.shape[0]
    target_edges = avg_degree * num_nodes // 2

    selected_rows = rows[:target_edges]
    selected_cols = cols[:target_edges]
    selected_values = values[:target_edges]

    edge_index = np.vstack([np.concatenate([selected_rows, selected_cols]),
                            np.concatenate([selected_cols, selected_rows])])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_weights = np.concatenate([selected_values, selected_values])
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)


    num_train = len(y_train)
    num_total = len(y)
    train_mask = torch.zeros(num_total, dtype=torch.bool)
    test_mask = torch.zeros(num_total, dtype=torch.bool)
    train_mask[:num_train] = True
    test_mask[num_train:] = True

    # 构建图数据对象
    data = Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_weight,
                y=torch.tensor(y, dtype=torch.long),
                train_mask=train_mask,
                test_mask=test_mask)

    return data




