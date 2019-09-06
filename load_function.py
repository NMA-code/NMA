# -*- coding=utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import random
from tqdm import tqdm
import os
import sys
import math
import networkx as nx
import scipy.sparse as ssp
from gensim.models import word2vec
import node2vec
import warnings

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/node2vec/src' % cur_dir)


def sample_neg(net, test_ratio=0.9, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg


def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None,
                    node_information=None):

    def helper(A, links, g_label):
        g_list = []
        for i, j in tqdm(list(zip(links[0], links[1]))):
            node_features, adj, nodes_weights, distance_weights = subgraph_extraction_labeling((i, j), A, h,
                                                                                               max_nodes_per_hop,
                                                                                               node_information)
            g_list.append((g_label, node_features, adj, nodes_weights, distance_weights))
        return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    return train_graphs, test_graphs


def make_new_labels(labels, num):
    new_labels = []
    for i in labels:
        if i < num:
            new_labels.append(i)
        else:
            new_labels.append(0)
    return new_labels


def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])

    nodes_dist = [0, 0]
    for dist in range(1, h + 1):
        fringe = neighbors(fringe, A[0][1])
        fringe = fringe - visited
        visited = visited.union(fringe)

        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    if A[0][0] > 30000:
        if len(nodes) > max_nodes_per_hop - 2:
            nodes = random.sample(nodes, max_nodes_per_hop - 2)
        nodes = [ind[0], ind[1]] + list(nodes)
        sub_graph = A[0][1][nodes, :][:, nodes]
        labels = node_label(sub_graph, max_nodes_per_hop + 1).tolist()
        if max_nodes_per_hop + 1 in labels:
            labels = make_new_labels(labels, max_nodes_per_hop + 1)
    else:
        nodes = [ind[0], ind[1]] + list(nodes)
        sub_graph = A[0][1][nodes, :][:, nodes]
        labels = node_label(sub_graph, max_nodes_per_hop + 1)
        node_labels = dict(sorted(zip(nodes, labels), key=lambda x: x[1]))
        nodes = list(node_labels.keys())
        labels = list(node_labels.values())
        if max_nodes_per_hop + 1 in labels:
            labels = make_new_labels(labels, max_nodes_per_hop + 1)
    if max_nodes_per_hop is not None and max_nodes_per_hop < len(nodes):
        nodes = nodes[:max_nodes_per_hop]
        labels = labels[:max_nodes_per_hop]

    sub_graph = A[0][1][nodes, :][:, nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(sub_graph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)

    if node_information is not None:
        features = torch.zeros((len(A), max_nodes_per_hop, len(node_information[0][0])))
    else:
        features = torch.zeros((len(A), max_nodes_per_hop, max_nodes_per_hop))
    adj = torch.zeros((len(A), max_nodes_per_hop, max_nodes_per_hop))
    nodes_weights = torch.zeros((len(A), 1, max_nodes_per_hop))
    distance_weights = torch.zeros((len(A), 1, max_nodes_per_hop))
    for i in range(len(A)):
        if node_information is not None:
            temp_x = torch.Tensor(node_information[i][nodes])
            if len(nodes) < max_nodes_per_hop:
                zero_vec_features_x = torch.zeros(max_nodes_per_hop - len(nodes), temp_x.shape[1])
                temp_x = torch.cat((temp_x, zero_vec_features_x), dim=0)
        else:
            temp_x = torch.Tensor(np.eye(len(nodes)))
            if len(nodes) < max_nodes_per_hop:
                zero_vec_features_x = torch.zeros(max_nodes_per_hop - len(nodes), temp_x.shape[1])
                zero_vec_features_y = torch.zeros(max_nodes_per_hop, max_nodes_per_hop - temp_x.shape[1])
                temp_x = torch.cat((temp_x, zero_vec_features_x), dim=0)
                temp_x = torch.cat((temp_x, zero_vec_features_y), dim=1)
        temp_adj = A[i][1][nodes, :][:, nodes]
        g_ = nx.from_scipy_sparse_matrix(temp_adj)
        labels_every = node_label(temp_adj, max_nodes_per_hop + 100).tolist()
        labels_every = make_new_labels(labels_every, max_nodes_per_hop + 1)
        temp_adj = torch.Tensor(temp_adj.todense())
        node_deg = list(dict(g_.degree).values())
        if i == 0:
            node_deg = list(dict(g.degree).values())
            temp_adj = torch.Tensor(nx.to_numpy_matrix(g))
            labels_every = labels
        if len(nodes) < max_nodes_per_hop:
            zero_vec_x = torch.zeros((max_nodes_per_hop - len(nodes), len(nodes)))
            zero_vec_y = torch.zeros((max_nodes_per_hop, max_nodes_per_hop - len(nodes)))
            temp_adj = torch.cat((temp_adj, zero_vec_x), dim=0)
            temp_adj = torch.cat((temp_adj, zero_vec_y), dim=1)
            node_deg = np.append(node_deg, np.zeros((1, max_nodes_per_hop - len(nodes))))
            zero_list = np.zeros((max_nodes_per_hop - len(nodes))).tolist()
            labels_every.extend(zero_list)
        node_deg_matrix = torch.Tensor(np.linalg.pinv(np.diag(node_deg)))
        nodes_weights[i] = torch.Tensor(node_deg).unsqueeze(-2)
        distance_weights[i] = torch.Tensor(labels_every).unsqueeze(-2)
        features[i] = temp_x
        adj[i] = torch.mm(node_deg_matrix, temp_adj) + node_deg_matrix
    return features, adj, nodes_weights, distance_weights


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph, max_num):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0] + [i for i in range(2, K)], :][:, [0] + [i for i in range(2, K)]]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = max_num
    labels[labels > 1e6] = max_num  # set inf labels to 0
    labels[labels < -1e6] = max_num  # set -inf labels to 0
    return labels


def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [str(walk) for walk in walks]
    model = word2vec.Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings


def get_adj_and_degs(graphs, layer_info, node_info, max_num):
    tmp_adj = torch.zeros((len(graphs), len(layer_info), max_num, max_num))
    tmp_labels = torch.LongTensor(len(graphs))

    if node_info is not None:
        tmp_x = torch.zeros(len(graphs), len(layer_info), max_num, len(node_info[0][0]))
    else:
        tmp_x = torch.zeros(len(graphs), len(layer_info), max_num, max_num)
    tmp_nodes_weights = torch.zeros((len(graphs), len(layer_info), 1, max_num))
    tmp_distance_weights = torch.zeros((len(graphs), len(layer_info), 1, max_num))

    for i in range(len(graphs)):
        lines = graphs[i]
        tmp_labels[i] = lines[0]
        tmp_x[i] = lines[1]
        tmp_adj[i] = lines[2]
        tmp_nodes_weights[i] = lines[3]
        tmp_distance_weights[i] = lines[4]

    return tmp_adj, tmp_x, tmp_nodes_weights, tmp_distance_weights, tmp_labels


def accuracy(output, labels):
    correct = output.eq(labels).double()
    correct = correct.sum()
    return correct * 100 / len(labels)
