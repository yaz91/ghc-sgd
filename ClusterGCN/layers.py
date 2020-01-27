import math
import torch
from torch.nn.modules.module import Module
import torch.nn as nn

class GraphConvolution(Module):


    def __init__(self, in_features, out_features, activation, dropout, bias=True, use_lynorm=True, precalc=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precalc = precalc

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.

        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_features, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):

        if self.precalc:
            h = input
        else:
            support = torch.mm(adj, input)
            h = torch.cat((input, support), dim=1)

        if self.dropout:
            h = self.dropout(h)

        h = self.linear(h)
        h = self.lynorm(h)

        if self.activation:
            h = self.activation(h)

        return h


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
