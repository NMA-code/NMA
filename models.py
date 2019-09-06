# -*- coding=utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphConvolution


class NMA(nn.Module):
    def __init__(self, n_feat, n_hid, n_nums, dropout, alpha, n_heads, n_gcn):
        """Dense version of GAT."""
        super(NMA, self).__init__()
        self.L = 2*n_heads*n_hid
        self.D = n_feat
        self.K = 1
        self.n_nums = n_nums
        self.n_heads = n_heads
        self.dropout = dropout
        self.GCN = GraphConvolution(n_feat, n_feat, n_gcn)

        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.attention_degs = nn.Sequential(
            nn.Linear(self.n_nums, self.n_nums),
            nn.ReLU()
        )

        self.attention_distance = nn.Sequential(
            nn.Linear(self.n_nums, self.n_nums),
            nn.ReLU()
        )

        self.attention_interlayer = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Linear(self.L, 2),
        )

    def forward(self, x, adj, degs_weights, distance_weights):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.GCN(x, adj)

        degs_weights = self.attention_degs(degs_weights)
        distance_weights = self.attention_distance(distance_weights)
        nodes_weights = torch.mul(degs_weights, distance_weights)
        nodes_weights = torch.transpose(nodes_weights, 3, 2)
        nodes_weights = nodes_weights.expand_as(x)
        x = torch.mul(x, nodes_weights)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=3)
        x = x[:, :, 0:2, :]
        x = x.view(x.shape[0], x.shape[1], 1, -1).squeeze(2)
        x = F.dropout(x, self.dropout, training=self.training)
        A = self.attention_interlayer(x)
        A = torch.transpose(A, 2, 1)
        A = F.softmax(A, dim=2)
        x = torch.matmul(A, x)
        logits = self.classifier(x.squeeze(1))
        return logits

