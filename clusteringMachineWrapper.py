from torch.utils.data import Dataset, DataLoader
import torch

class clusteringMachineWrapper(Dataset):
    def __init__(self,args, ClusteringMachine):
        self.ClusteringMachine = ClusteringMachine
        self.featureCount = self.ClusteringMachine.feature_count
        self.classCount = self.ClusteringMachine.class_count
        self.targets = [self.ClusteringMachine.sg_targets[cluster] for cluster in self.ClusteringMachine.clusters]
        self.data = []
        for cluster in self.ClusteringMachine.clusters:
            self.data.append({'nodes':self.ClusteringMachine.sg_nodes[cluster],
               'edges':self.ClusteringMachine.sg_edges[cluster],
               'train':self.ClusteringMachine.sg_train_nodes[cluster],
               'test':self.ClusteringMachine.sg_test_nodes[cluster],
               'features':self.ClusteringMachine.sg_features[cluster]})
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx],self.targets[idx]
#     {'node':self.ClusteringMachine.sg_node[idx],
#                'edge':self.ClusteringMachine.sg_nodes[idx],
#                'train':self.ClusteringMachine.sg_train_nodes[idx],
#                'test':self.ClusteringMachine.sg_test_nodes[idx],
#                'features':self.ClusteringMachine.sg_features[idx],
#                'targets':self.ClusteringMachine.sg_targets[idx]}
                
    def __len__(self):
         return len(self.ClusteringMachine.clusters)