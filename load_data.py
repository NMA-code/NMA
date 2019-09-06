# -*- coding=utf-8 -*-
from scipy.sparse import coo_matrix
import numpy as np
import os


def load_nodes_num(file_name):
    file_name = file_name + 'nodes.txt'
    with open(file_name, 'r') as f:
        data = f.readlines()
        nodes_num = len(data)
    return nodes_num


def get_max_label(file_name):
    data = np.loadtxt(file_name + 'edges.txt')
    if data.shape[1] < 4:
        _, i, j,  = data.T
    else:
        _, i, j, _ = data.T
    max_i = np.max(i)
    max_j = np.max(j)
    return int(max(max_i, max_j)) + 1


def load_data(file_name):
    file_name_new = file_name + 'edges.txt'
    if not os.path.exists(file_name + 'nodes.txt'):
        num = get_max_label(file_name)
    else:
        num = load_nodes_num(file_name)
    layer_in = []
    with open(file_name_new, 'r+') as f:
        data = f.readlines()
        layer_label = []
        lines_num = 0
        matrix_layer = []
        for lines in data:
            items = list(map(float, lines.strip().split(' ')))
            if lines_num == 0:
                layer_label.append(items[0])
                matrix_layer.append([items[1], items[2]])
                lines_num += 1
            else:
                if items[0] in layer_label:
                    matrix_layer.append([items[1], items[2]])
                    lines_num += 1
                else:
                    i, j = np.array(matrix_layer).T
                    adj = coo_matrix((np.ones(i.shape[0]), (i.astype(int), j.astype(int))), shape=(num, num), dtype=np.float64)
                    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
                    layer_in.append((lines_num, adj.tocsr()))
                    lines_num = 1
                    matrix_layer.clear()
                    matrix_layer.append([items[1], items[2]])
                    layer_label.append(items[0])
        i, j = np.array(matrix_layer).T
        adj = coo_matrix((np.ones(i.shape[0]), (i.astype(int), j.astype(int))), shape=(num, num), dtype=np.float64)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        layer_in.append((lines_num, adj.tocsr()))
    return layer_in

