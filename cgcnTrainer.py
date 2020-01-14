'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import torch.distributed as dist
from myutils import progress_bar
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

class Average(object):

    def all_reduce(self):
        value = torch.Tensor([self.sum, self.count]).cuda()/dist.get_world_size()
        dist.all_reduce(value)
        self.sum, self.count = value[0], value[1]
        return self

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        # self.all_reduce()
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number
        return self


class Accuracy(object):

    def __init__(self):
        self.correct = 0.0
        self.count = 0.0

    def all_reduce(self):
        value = torch.Tensor([self.correct, self.count]).cuda()/dist.get_world_size()
        dist.all_reduce(value)
        self.correct, self.count = value[0], value[1]
        return self

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count*100

    def update(self, output, target):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)
        return self


class cgcnTrainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.record = [] 

    def adjust_learning_rate(self, epoch, args):
        lr = args.lr * (0.1 ** (epoch // 50))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def fit(self, best_acc, start_epoch, epochs, args):
        record = []
        train_loss, train_acc = self.evaluate(0, args, False)
        test_loss, test_acc = self.evaluate(0, args)
        record.append([float(train_loss), float(train_acc), float(test_loss), float(test_acc)])
        for epoch in range(start_epoch+1, start_epoch+epochs + 1):
            train_loss, _ = self.train(epoch, args, False)
            train_loss, train_acc = self.evaluate(epoch, args, False)
            test_loss, test_acc = self.evaluate(epoch, args)
            acc = 0
            if args.rank==0 and epoch == start_epoch+epochs:
                    # print('Saving..')
                    state = {
                        'net': self.model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, './checkpoint/'+args.model+".pth")
            if acc > best_acc:
                if args.rank==0:
                    print('Saving..')
                    state = {
                        'net': self.model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, './checkpoint/'+args.model+".pth")
                best_acc = acc
            record.append([float(train_loss), float(train_acc), float(test_loss), float(test_acc)])
            if args.rank == 0 and epoch%10==0:
                record_csv = pd.DataFrame(record)
                record_csv.to_csv(self.get_filename(args)+"_"+str(epoch)+".csv", index=False, header=None)
        
    def get_filename(self, args):
        alg_name = "minisgd"
        if args.period > 1 and args.vrl and args.ghc:
            alg_name = "gsgd"
        elif args.period > 1 and args.vrl:
            alg_name = "vrlsgd"
        elif args.period > 1:
            alg_name = "localsgd"
        filename = "record/{}_{}_{}_local_{}_vrl_{}_eta_{}_mom_{}_b_{}_cluster_data_{}".format(alg_name, args.model, args.dataset, args.period, int(args.vrl) , args.lr,args.momentum, args.batch_size, args.cluster_data)
        return filename

    def warm_up(self, optimizer, lr_grow):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] + lr_grow

    def train(self, epoch, args, fake_acc=False):
        self.model.train()
        # self.model.eval()
        if args.rank==0:
            print('\nEpoch: %d' % epoch)
            
        world_size = args.world_size
        target_set = []
        data_set = []
        update_cnt = 0
        train_loss = 0
        train_acc = 0
        self.node_count_seen = 0
        self.accumulated_training_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            
            batch_average_loss, node_count = self.do_forward_pass(data, target)
            batch_average_loss.backward()
            self.optimizer.step()
            train_loss = self.update_average_loss(batch_average_loss, node_count)
            
            if args.rank ==0:
                progress_bar(batch_idx, len(self.train_loader), 'Loss: %.5f'
                    % (train_loss), fake_acc=fake_acc)

        self.model._sync_period()
        return train_loss, train_acc

    def evaluate(self, epoch, args, use_test=True):
        # print('\nEpoch: %d' % epoch)
        self.model.eval()
        self.predictions = []
        self.targets = []
        self.accumulated_training_loss = 0
        self.node_count_seen = 0
        if use_test:
            test_loader = self.test_loader
        else:
            test_loader = self.train_loader
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                prediction, target,batch_average_loss,node_count = self.do_prediction(data,target, use_test=use_test)
                self.predictions.append(prediction.cpu().detach().numpy())
                self.targets.append(target.cpu().detach().numpy())
                test_loss = self.update_average_loss(batch_average_loss, node_count)
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)
        test_acc = f1_score(self.targets, self.predictions, average="micro")
        return test_loss, test_acc
    
    def do_prediction(self, data,target, use_test=True):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = data['edges'][0].to(self.device)
        macro_nodes = data['nodes'][0].to(self.device)
        if use_test:
            test_nodes = data['test'][0].to(self.device)
        else:
            test_nodes = data['train'][0].to(self.device)
        features = data['features'][0].to(self.device)
        target = target.to(self.device).squeeze()[test_nodes]
        predictions = self.model(edges, features)
        prediction = predictions[test_nodes,:]
        node_count = test_nodes.shape[0]
        average_loss = torch.nn.functional.nll_loss(prediction, target)
        return prediction, target,average_loss,node_count
    
    def do_forward_pass(self,data,target):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = data['edges'][0].to(self.device)
        macro_nodes = data['nodes'][0].to(self.device)
        train_nodes = data['train'][0].to(self.device)
        features = data['features'][0].to(self.device)
        target = target.to(self.device).squeeze()
        predictions = self.model(edges, features)
        average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        return average_loss