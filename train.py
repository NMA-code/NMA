# -*- coding=utf-8 -*-
from __future__ import division
from __future__ import print_function

import time
import argparse
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve)

from load_function import *
from load_data import load_data
from models import NMA

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='CKM-Physicians-Innovation_Multiplex_Social', help='network name')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--max_nodes_per_hop', type=int, default=25, help='max num of neighbors.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=10, help='Number of head attentions.')
parser.add_argument('--n_gcn', type=int, default=2, help='Number of gcn layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.01, help='Alpha for the leaky_relu.')
parser.add_argument('--batch', type=float, default=100, help='batch size.')
parser.add_argument('--features_num', type=int, default=128, help='batch size.')
parser.add_argument('--hop', default=1, type=int, help='enclosing subgraph hop number, options: 1, 2,..., ')
parser.add_argument('--k_nums', type=int, default=10, help='k numbers of K-fold cross validation.')
parser.add_argument('--use_embedding', type=bool, default=False, help='whether to use node2vec node embeddings.')
parser.add_argument('--patience', type=int, default=20, help='Patience')
parser.add_argument('--max_train_num', type=int, default=20000, help='set maximum number of train (to fit into memory)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

# Model and optimizer
if args.use_embedding:
    model = NMA(n_feat=args.features_num,
                n_hid=args.hidden,
                n_nums=args.max_nodes_per_hop,
                dropout=args.dropout,
                n_heads=args.nb_heads,
                n_gcn=args.n_gcn,
                alpha=args.alpha)
else:
    model = NMA(n_feat=args.max_nodes_per_hop,
                n_hid=args.hidden,
                n_nums=args.max_nodes_per_hop,
                dropout=args.dropout,
                n_heads=args.nb_heads,
                n_gcn=args.n_gcn,
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
criteon = nn.CrossEntropyLoss().cuda()

if args.cuda:
    model.cuda()
    print('\nGPU is ON!')


def loop_dataset(graphs, models, sample_idxes, optimizer=None, bsize=None, layers_info=None, node_information=None):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_pred = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [graphs[idx] for idx in selected_idx]
        adj, x, degs_weights, distance_weights, labels = get_adj_and_degs(batch_graph, layers_info, node_information,
                                                                          args.max_nodes_per_hop)
        if args.cuda:
            adj = adj.cuda()
            x = x.cuda()
            degs_weights = degs_weights.cuda()
            distance_weights = distance_weights.cuda()
            labels = labels.cuda()
        all_targets += labels.cpu().numpy().tolist()
        logits = models(x, adj, degs_weights, distance_weights)
        loss = criteon(logits, labels)

        preds = logits.max(1)[1].type_as(labels)
        all_pred += preds.cpu().numpy().tolist()

        acc = accuracy(preds, labels)
        all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    all_targets = np.array(all_targets)

    all_pred = np.array(all_pred)

    ps, rs, _ = precision_recall_curve(all_targets, all_scores)
    auc_ = auc(rs, ps)
    avg_loss = np.concatenate((avg_loss, [auc_], [roc_auc_score(all_targets, all_scores)], [f1_score(all_targets, all_pred)]))

    return avg_loss


def train(epoch_, train_graphs, test_graphs, layers_info, node_information):
    t = time.time()
    model.train()
    train_index = list(range(len(train_graphs)))

    # train
    ave_loss_train = loop_dataset(train_graphs, model, train_index, optimizer=optimizer, bsize=args.batch,
                                  layers_info=layers_info, node_information=node_information)

    # test
    model.eval()
    ave_loss_test = loop_dataset(test_graphs, model, list(range(len(test_graphs))), bsize=args.batch,
                                 layers_info=layers_info, node_information=node_information)

    end = time.time()
    print("\033[93m Epoch {}:\033[0m".format(epoch_ + 1),
          "\033[93m train_loss: {:.5f}\033[0m".format(ave_loss_train[0]),
          "\033[93m test_loss= {:.5f}\033[0m".format(ave_loss_test[0]),
          "\033[93m train_accuracy= {:.5f}%\033[0m".format(ave_loss_train[1]),
          "\033[93m test_accuracy= {:.5f}%\033[0m".format(ave_loss_test[1]),
          '\033[93m time: {:.5f}s \033[0m'.format(end - t))
    return ave_loss_test


def main():
    # Load data
    file_name = os.path.join(args.file_dir, 'data/{}/'.format(args.data_name))
    layers_info = load_data(file_name)
    train_pos, train_neg, test_pos, test_neg = sample_neg(layers_info[0][1], test_ratio=0.1, max_train_num=args.max_train_num)

    node_information = None
    node_information_ = []
    if args.use_embedding:
        for l in range(0, len(layers_info)):
            node_information_.append(generate_node2vec_embeddings(layers_info[l][1], args.features_num))

    if len(node_information_) != 0:
        node_information = node_information_

    '''Train and apply classifier'''
    # A = layers_info[0][1].copy()  # the observed network
    layers_info[0][1][test_pos[0], test_pos[1]] = 0  # mask test links
    layers_info[0][1][test_pos[1], test_pos[0]] = 0  # mask test links
    # layers_info[0][1] = A.copy()

    train_graphs, test_graphs = links2subgraphs(layers_info, train_pos, train_neg, test_pos, test_neg, args.hop,
                                                args.max_nodes_per_hop, node_information=node_information)
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    # mess up the order
    random.shuffle(train_graphs)
    random.shuffle(test_graphs)

    epoch_test_acc = []
    epoch_test_pr = []
    epoch_test_auc = []
    epoch_test_f1 = []

    best_loss = args.epochs + 1
    bad_counter = 0

    for epoch in range(args.epochs):
        acc_and_auc = train(epoch, train_graphs, test_graphs, layers_info, node_information)
        if acc_and_auc[0] < best_loss:
            best_loss = acc_and_auc[0]
            bad_counter = 0
            epoch_test_pr.append(acc_and_auc[2])
            epoch_test_auc.append(acc_and_auc[3])
            epoch_test_f1.append(acc_and_auc[4])
            epoch_test_acc.append(round(acc_and_auc[1].item()/100, 5))

        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    over_test_res = [np.mean(np.array(epoch_test_acc)), np.mean(np.array(epoch_test_auc)),
                     np.mean(np.array(epoch_test_f1)), np.mean(np.array(epoch_test_pr))]
    return over_test_res


if __name__ == '__main__':
    over_test_acc = []
    over_test_pr_auc = []
    over_test_roc_auc = []
    over_test_f1 = []
    num = args.k_nums + 1
    for i in range(num):
        print()
        print('This is {}th fold valid'.format(i + 1))
        print()
        print('This is data of {}'.format(args.data_name))
        print()

        res = main()
        if i != 0:
            over_test_acc.append(round(res[0], 5))
            over_test_pr_auc.append(round(res[3], 5))
            over_test_roc_auc.append(round(res[1], 5))
            over_test_f1.append(round(res[2], 5))
    over_test_acc.append(np.mean(over_test_acc))
    over_test_pr_auc.append(np.mean(over_test_pr_auc))
    over_test_roc_auc.append(np.mean(over_test_roc_auc))
    over_test_f1.append(np.mean(over_test_f1))

    if args.use_embedding:
        filename = 'embedding_'
    else:
        filename = 'no_embedding_'
    with open(os.path.join(args.file_dir, 'data/{}//results/{}results_res.txt'.format(args.data_name, filename)), 'a+')\
            as f:
        f.write(str(args) + '\n')
        f.write('acc:' + ' '.join(map(str, over_test_acc)) + '\n')
        f.write('pr_auc:' + ' '.join(map(str, over_test_pr_auc)) + '\n')
        f.write('roc_auc:' + ' '.join(map(str, over_test_roc_auc)) + '\n')
        f.write('f1:' + ' '.join(map(str, over_test_f1)) + '\n')

