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
import torch.multiprocessing as mp
import os
import argparse
from myutils import *
from trainer import *
from torch import distributed, nn
from restarted import DistributedDataParallel
from DistributedSgd import DSGD
from UnshuffleSampler import UnshuffleDistributedSampler
from graphUnshuffleSampler import graphUnshuffleDistributedSampler

from cgcnTrainer import cgcnTrainer

import ClusterGCN.graph_iter as graphIter
import ClusterGCN.utils as ghcUtils
from ClusterGCN.models import StackedGCN as sgcn
from sgcnTrainer import sgcnTrainer
from clusterUnshuffleSampler import clusterUnshuffleDistributedSampler

def work_process(gpu, ngpus_per_node, args, clusterMachine):
    args.rank = args.rank * ngpus_per_node + gpu 
    torch.cuda.set_device(args.rank+args.st) 
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )
    device = torch.device("cuda:"+str(args.rank))
    if (args.model != "cgcn") & (args.model != "clustergcn"):
        model = get_model(args)
        model = DistributedDataParallel(model, device_ids=[args.rank+args.st], local=args.local, update_period=args.period)
        cudnn.benchmark = True
        best_acc = 0  # best test accuracy
        start_epoch = 0 
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/'+args.model+"_init.pth")
            model.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        optimizer = DSGD(model.parameters(), local=args.local, update_period=args.period, model=model, lr=args.lr, momentum=0, weight_decay=1e-4, vrl=args.vrl)

        train_dataset, test_dataset = get_dataset(args.dataset, args)
        val_dataset, test_dataset = get_dataset(args.dataset, args)

        train_sampler = UnshuffleDistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, cluster_data=args.cluster_data)
        val_loader = None
        val_loader = data.DataLoader(val_dataset, args.batch_size*args.world_size, shuffle=True)
        train_loader = data.DataLoader(train_dataset, args.batch_size, shuffle= (train_sampler is None),sampler=train_sampler)
        test_loader = data.DataLoader(test_dataset, args.batch_size*args.world_size, shuffle=True, num_workers=2)

        trainer = Trainer(model, optimizer, train_loader, test_loader, device, val_loader)
        trainer.fit(best_acc, start_epoch, args.epochs, args)
    elif args.model == "cgcn":
        dataset = get_dataset(args.dataset, args,clueterMachine)
        train_sampler = graphUnshuffleDistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, cluster_data=args.cluster_data)
        val_loader = data.DataLoader(dataset, 1, shuffle=True)
        train_loader = data.DataLoader(dataset, 1, shuffle= (train_sampler is None),sampler=train_sampler)
        test_loader = data.DataLoader(dataset, 1, shuffle=True, num_workers=1)

        model = get_model(args,10,dataset.featureCount,dataset.classCount)
        model = DistributedDataParallel(model, device_ids=[args.rank+args.st], local=args.local, update_period=args.period)
        cudnn.benchmark = True
        best_acc = 0  # best test accuracy
        start_epoch = 0 
        
        optimizer = DSGD(model.parameters(), local=args.local, update_period=args.period, model=model, lr=args.lr, momentum=args.momentum, weight_decay=1e-4, vrl=args.vrl)
        
        trainer = cgcnTrainer(model, optimizer, train_loader, test_loader, device, val_loader)
        trainer.fit(best_acc, start_epoch, args.epochs, args)
    elif args.model == "clustergcn":
        train_sampler = clusterUnshuffleDistributedSampler(clusterMachine['train'], num_replicas=args.world_size, rank=args.rank, cluster_data=args.cluster_data)
        train_loader = data.DataLoader(clusterMachine['train'], 1, shuffle=(train_sampler is None), num_workers=2,sampler=train_sampler)
        val_loader = data.DataLoader(clusterMachine['val'], 1, shuffle=True, num_workers=2)
        test_loader = data.DataLoader(clusterMachine['test'], 1, shuffle=True, num_workers=2)
        
        model = sgcn(args, clusterMachine['in'], clusterMachine['out'], args.dropout, bias=True, use_lynorm=True, precalc=False).to(device)
        model = DistributedDataParallel(model, device_ids=[args.rank+args.st], local=args.local, update_period=args.period)
        
        cudnn.benchmark = True
        best_acc = 0  # best test accuracy
        start_epoch = 0 
        
        if args.multilabel:
            print('Using multi-label loss')
            loss_f = nn.BCEWithLogitsLoss()
        else:
            print('Using multi-class loss')
            loss_f = nn.CrossEntropyLoss()
    
        optimizer = DSGD(model.parameters(), local=args.local, update_period=args.period, model=model, lr=args.lr, momentum=args.momentum, weight_decay=1e-4, vrl=args.vrl)
        
        trainer = sgcnTrainer(model,loss_f, optimizer, train_loader, test_loader, device, val_loader,args.multilabel)
        trainer.fit(best_acc, start_epoch, args.epochs, args)
        

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.00, type=float, help='momentum')
    parser.add_argument('--model', type=str, default='vgg16', help='Name of the model to use.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset to use.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--root', type=str, default='./data')
    
    parser.add_argument("--edge-path",nargs = "?",default = "./data/graphData/edges.csv",help = "Edge list csv.")
    parser.add_argument("--features-path",nargs = "?",default = "./data/graphData/features.csv",help = "Features json.")
    parser.add_argument("--target-path",nargs = "?",default = "./data/graphData/target.csv",help = "Target classes csv.")
    parser.add_argument("--clustering-method",nargs = "?",default = "metis",help = "Clustering method for graph decomposition. Default is the metis procedure.")
    parser.add_argument("--seed",type = int,default = 42,help = "Random seed for train-test split. Default is 42.")    
    parser.add_argument("--dropout",type = float,default = 0.5,help = "Dropout parameter. Default is 0.5.")
    parser.add_argument("--learning-rate",type = float,default = 0.01,help = "Learning rate. Default is 0.01.")
    parser.add_argument("--test-ratio",type = float,default = 0.9,help = "Test data ratio. Default is 0.1.")
    parser.add_argument("--cluster-number",type = int,default = 40,help = "Number of clusters extracted. Default is 10.")
#     parser.set_defaults(layers = [16, 16, 16])   

    parser.add_argument('--layers', nargs='+', type=int, default=[2048, 2048, 2048]) 
    
    parser.add_argument("--data-prefix",nargs = "?",default = "/home/yaz91/data",help = "data path.")
    parser.add_argument('--precalc', action='store_false', default=True, help='Whether to pre-calculate the first layer (AX preprocessing).')
    parser.add_argument('--num_clusters', type=int, default=50, help='Number of clusters.')
    parser.add_argument('--num_clusters_val', type=int, default=50, help='Number of clusters for validation.')
    parser.add_argument('--num_clusters_test', type=int, default=50, help='Number of clusters for test.')
    parser.add_argument('--diag_lambda', type=float, default=1, help='A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement.')
    parser.add_argument('--bsize', type=int, default=1, help='Number of clusters for each batch.')
    parser.add_argument('--multilabel', action='store_false', default=True, help='Multilabel or multiclass.')
    
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('--period', default=10, type=int, help='update period')
    parser.add_argument('--st', default=0, type=int, help='gpu st')
    parser.add_argument('--port', default=23456, type=int, help='port')
    parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu-num', default=4, type=int, help='gpu num')
    parser.add_argument('-v', '--vrl', dest='vrl', action='store_true')
    parser.set_defaults(vrl=False)
    parser.add_argument('--ghc', dest='ghc', action='store_true')
    parser.set_defaults(ghc=False)
    parser.add_argument('--local', dest='local', action='store_true')
    parser.set_defaults(local=False)
    parser.add_argument('--cluster-data', dest='cluster_data', action='store_true')
    parser.set_defaults(cluster_data=False)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--backend', type=str, default='nccl', help='Name of the backend to use.')
    args = parser.parse_args()
    if not args.local:
        args.period = 1
    print(args)
    args.init_method = 'tcp://127.0.0.1:'+str(args.port)
    ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = args.gpu_num
    args.world_size = ngpus_per_node * args.world_size
    args.batch_size = int(args.batch_size//args.world_size)
    if args.dataset == "graphData":
        graph = graph_reader(args.edge_path)
        features = feature_reader(args.features_path)
        target = target_reader(args.target_path)
        clusterMachine = ClusteringMachine(args, graph, features, target)
        clusterMachine.decompose()
    elif args.dataset == "ppi":
        clusterMachine = dict()
        (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
         train_mask, val_mask, test_mask, train_data, val_data, test_data,
         num_data, visible_data) = ghcUtils.load_data(args.data_prefix, args.dataset, args.precalc)
        
        train_dataset = graphIter.GraphData(train_adj, train_feats, y_train, train_mask, visible_data,
                                             args.num_clusters, args.bsize, args.diag_lambda)
        val_dataset = graphIter.GraphData(full_adj, test_feats, y_val, val_mask, np.arange(num_data),
                                             args.num_clusters_val, 1, args.diag_lambda)
        test_dataset = graphIter.GraphData(full_adj, test_feats, y_test, test_mask, np.arange(num_data),
                                             args.num_clusters_test, 1, args.diag_lambda)

        inputFeature = train_feats.shape[1]
        outputFeature = y_train.shape[1]
        clusterMachine.update({'train':train_dataset,'val':val_dataset,'test':test_dataset,'in':inputFeature,'out':outputFeature})
        
    mp.spawn(work_process, nprocs=ngpus_per_node, args=(ngpus_per_node, args, clusterMachine))

if __name__ == "__main__":
    main()