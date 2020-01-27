import torch
# from torch_geometric.nn import GCNConv
from ClusterGCN.layers import GraphConvolution
import torch.nn.functional as F

class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(self, args, input_features, output_features, dropout, bias=True, use_lynorm=True, precalc=False):
        super(StackedGCN, self).__init__()
        self.args = args

        self.setup_layers(input_features, output_features, dropout, bias, use_lynorm, precalc)

    def setup_layers(self, input_features, output_features, dropout, bias, use_lynorm, precalc):
        self.layers = []

        self.layers.append(GraphConvolution(input_features if precalc else input_features * 2,
                                            self.args.layers[0],
                                            activation=F.relu, dropout=dropout, bias=bias, use_lynorm=use_lynorm, precalc=precalc))

        for i, _ in enumerate(self.args.layers[:-2]):
            self.layers.append(GraphConvolution(self.args.layers[i] * 2,
                                                self.args.layers[i + 1],
                                                activation=F.relu, dropout=dropout, bias=bias, use_lynorm=use_lynorm, precalc=False))

        self.layers.append(GraphConvolution(self.args.layers[-1] * 2,
                                            output_features,
                                            activation=lambda x: x, dropout=dropout, bias=bias, use_lynorm=False, precalc=False))


        self.layers = ListModule(*self.layers)

    def forward(self, adj, features):
        for i, _ in enumerate(self.layers):
            features = self.layers[i](features, adj)

        return features

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)
