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


import ClusterGCN.utils as ghcUtils

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


class sgcnTrainer(object):

    def __init__(self, model,loss_f, optimizer, train_loader, test_loader, device, val_loader,multilabel):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.record = [] 
        self.loss_f = loss_f
        self.multilabel = multilabel

    def adjust_learning_rate(self, epoch, args):
        lr = args.lr * (0.1 ** (epoch // 50))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def fit(self, best_acc, start_epoch, epochs, args):
        record = []
        train_loss, train_micro, train_macro = self.evaluate(self.train_loader)
        val_loss, val_micro, val_macro = self.evaluate(self.val_loader)
        test_loss, test_micro, test_macro  = self.evaluate(self.test_loader)
        record.append([float(train_loss), float(train_micro), float(train_macro),
                       float(val_loss), float(val_micro), float(val_macro),
                       float(test_loss), float(test_micro), float(test_macro)])
        for epoch in range(start_epoch+1, start_epoch+epochs + 1):
            train_loss, _ = self.train(epoch, args, False)
            train_loss, train_micro, train_macro = self.evaluate(self.train_loader)
            val_loss, val_micro, val_macro = self.evaluate(self.val_loader)
            test_loss, test_micro, test_macro  = self.evaluate(self.test_loader)
            acc = val_micro
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
                    torch.save(state, './checkpoint/'+self.get_filename(args,short=True)+".pth")
#                     torch.save(state, './checkpoint/'+args.model+".pth")
                best_acc = acc
            record.append([float(train_loss), float(train_micro), float(train_macro),
                           float(val_loss), float(val_micro), float(val_macro),
                           float(test_loss), float(test_micro), float(test_macro)])
            if args.rank == 0 and epoch%10==0:
                record_csv = pd.DataFrame(record)
                record_csv.to_csv(self.get_filename(args)+".csv", index=False, header=None)
#                 record_csv.to_csv(self.get_filename(args)+"_"+str(epoch)+".csv", index=False, header=None)
        
    def get_filename(self, args,short=False):
        alg_name = "minisgd"
        if args.period > 1 and args.vrl and args.ghc:
            alg_name = "gsgd"
        elif args.period > 1 and args.vrl:
            alg_name = "vrlsgd"
        elif args.period > 1:
            alg_name = "localsgd"
        filename = "{}_{}_{}_local_{}_vrl_{}_eta_{}_mom_{}_b_{}_cluster_data_{}".format(alg_name, args.model, args.dataset, args.period, int(args.vrl) , args.lr,args.momentum, args.batch_size, args.cluster_data)
        if not short:
            filename = "record/"+filename            
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
        train_nodes = 0
        for batch_idx, (features_b, support_b, label_b, mask_b) in enumerate(self.train_loader):
            features_b = features_b[0].to(self.device)
            support_b = support_b[0].to(self.device)
            label_b = label_b[0].to(self.device)
            mask_b = mask_b[0].to(self.device)

            pred = self.model(support_b, features_b)
            loss = self.loss_f(pred[mask_b], label_b[mask_b])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()*features_b.shape[0]
            train_nodes += features_b.shape[0]
                        
            if args.rank ==0:
                progress_bar(batch_idx, len(self.train_loader), 'Loss: %.5f'
                    % (train_loss), fake_acc=fake_acc)

        self.model._sync_period()
        return train_loss/train_nodes, train_acc
    
    def evaluate(self,data_loader):
        self.model.eval()

        total_pred = []
        total_lab = []
        total_loss = 0
        total_nodes = 0

        for batch_idx, (features_b, support_b, label_b, mask_b) in enumerate(data_loader):
            num_data_b = np.sum(mask_b.cpu().detach().numpy())

            if num_data_b == 0:
                continue
            else:

                features_b = features_b[0].to(self.device)
                support_b = support_b[0].to(self.device)
                label_b = label_b[0].to(self.device)
                mask_b = mask_b[0].to(self.device)

                features_b = features_b.to(self.device)
                support_b = support_b.to(self.device)
                label_b = label_b.to(self.device)
                mask_b = mask_b.to(self.device)

                pred = self.model(support_b, features_b)
                loss = self.loss_f(pred[mask_b], label_b[mask_b])

                total_lab.append(label_b[mask_b].cpu().detach().numpy())
                total_pred.append(pred[mask_b].cpu().detach().numpy())
                total_loss += loss.cpu().detach().numpy() * num_data_b
                total_nodes += num_data_b

        total_pred = np.vstack(total_pred)
        total_lab = np.vstack(total_lab)


        micro, macro = ghcUtils.calc_f1(total_pred, total_lab, self.multilabel)

        return total_loss/total_nodes, micro, macro