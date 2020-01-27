import torch
import torch.utils.data
import ClusterGCN.utils as utils
import ClusterGCN.partition_utils as partition_utils
import numpy as np

class GraphData(torch.utils.data.Dataset):

    def __init__(self, adj, feats, label, mask, visible_data, num_clusters, bsize_for_train, diag_lambda=-1):
        super(GraphData, self).__init__()

        if bsize_for_train > 1:
            _, parts = partition_utils.partition_graph(adj, visible_data,
                                                       num_clusters)
            parts = [np.array(pt) for pt in parts]

            (self.features_batches, self.support_batches, self.label_batches,
             self.mask_batches) = utils.preprocess_multicluster(
                adj, parts, feats, label, mask,
                num_clusters, bsize_for_train, diag_lambda)
        else:
            (_, self.features_batches, self.support_batches, self.label_batches,
             self.mask_batches) = utils.preprocess(adj, feats, label,
                                                    mask, visible_data,
                                                    num_clusters,
                                                    diag_lambda)

        for index in range(len(self.features_batches)):
            self.features_batches[index] = torch.FloatTensor(self.features_batches[index])
            self.label_batches[index] = torch.FloatTensor(self.label_batches[index])
            if hasattr(torch, 'BoolTensor'):
                self.mask_batches[index] = torch.BoolTensor(self.mask_batches[index])
            else:
                self.mask_batches[index] = torch.ByteTensor(self.mask_batches[index])
            self.support_batches[index] = torch.FloatTensor(self.support_batches[index].todense())


            # sparse_mx = self.support_batches[index].tocoo().astype(np.float32)
            # indices = torch.from_numpy(
            #     np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            # values = torch.from_numpy(sparse_mx.data)
            # shape = torch.Size(sparse_mx.shape)
            # self.support_batches[index] = torch.sparse.FloatTensor(indices, values, shape)


    def __getitem__(self, index):
        features_b = self.features_batches[index]
        support_b = self.support_batches[index]
        label_b = self.label_batches[index]
        mask_b = self.mask_batches[index]

        return features_b, support_b, label_b, mask_b


    def __len__(self):
        return len(self.features_batches)
