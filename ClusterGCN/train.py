import utils
import argparse
import graph_iter
import numpy as np
from models import StackedGCN
import torch
from torch.utils import data
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--data_prefix', type=str, default='/home/yaz91/data', help='Datapath prefix.')
parser.add_argument('--dataset',  type=str, default='ppi', help='Dataset string.')
parser.add_argument('--precalc', action='store_false', default=True, help='Whether to pre-calculate the first layer (AX preprocessing).')
parser.add_argument('--num_clusters', type=int, default=50, help='Number of clusters.')
parser.add_argument('--num_clusters_val', type=int, default=5, help='Number of clusters for validation.')
parser.add_argument('--num_clusters_test', type=int, default=1, help='Number of clusters for test.')
parser.add_argument('--diag_lambda', type=float, default=1, help='A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement.')
parser.add_argument('--bsize', type=int, default=1, help='Number of clusters for each batch.')
parser.add_argument('--multilabel', action='store_false', default=True, help='Multilabel or multiclass.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument("--dropout", type = float, default=0.2, help="Dropout parameter. Default is 0.5.")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--layers', nargs='+', type=int, default=[2048, 2048, 2048])
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
 train_mask, val_mask, test_mask, train_data, val_data, test_data,
 num_data, visible_data) = utils.load_data(args.data_prefix, args.dataset, args.precalc)

train_dataset = graph_iter.GraphData(train_adj, train_feats, y_train, train_mask, visible_data,
                                     args.num_clusters, args.bsize, args.diag_lambda)
val_dataset = graph_iter.GraphData(full_adj, test_feats, y_val, val_mask, np.arange(num_data),
                                     args.num_clusters_val, 1, args.diag_lambda)
test_dataset = graph_iter.GraphData(full_adj, test_feats, y_test, test_mask, np.arange(num_data),
                                     args.num_clusters_test, 1, args.diag_lambda)

train_loader = data.DataLoader(train_dataset, 1, shuffle=True, num_workers=2)
val_loader = data.DataLoader(val_dataset, 1, shuffle=True, num_workers=2)
test_loader = data.DataLoader(test_dataset, 1, shuffle=True, num_workers=2)


print(train_feats.shape[1], y_train.shape[1])
model = StackedGCN(args, train_feats.shape[1], y_train.shape[1], args.dropout, bias=True, use_lynorm=True, precalc=False)
model = model.to(device)

if args.multilabel:
    print('Using multi-label loss')
    loss_f = nn.BCEWithLogitsLoss()
else:
    print('Using multi-class loss')
    loss_f = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def eval(data_loader):
    model.eval()

    total_pred = []
    total_lab = []
    total_loss = 0
    total_nodes = 0

    for batch_idx, (features_b, support_b, label_b, mask_b) in enumerate(data_loader):
        num_data_b = np.sum(mask_b.cpu().detach().numpy())

        if num_data_b == 0:
            continue
        else:

            features_b = features_b[0].to(device)
            support_b = support_b[0].to(device)
            label_b = label_b[0].to(device)
            mask_b = mask_b[0].to(device)

            features_b = features_b.to(device)
            support_b = support_b.to(device)
            label_b = label_b.to(device)
            mask_b = mask_b.to(device)

            pred = model(support_b, features_b)
            loss = loss_f(pred[mask_b], label_b[mask_b])

            total_lab.append(label_b[mask_b].cpu().detach().numpy())
            total_pred.append(pred[mask_b].cpu().detach().numpy())
            total_loss += loss.cpu().detach().numpy() * num_data_b
            total_nodes += num_data_b

    total_pred = np.vstack(total_pred)
    total_lab = np.vstack(total_lab)


    micro, macro = utils.calc_f1(total_pred, total_lab, args.multilabel)

    return total_loss/total_nodes, micro, macro




def train():


    for epoch in range(args.epochs):

        #========train===========
        model.train()
        train_loss = 0
        train_nodes = 0
        for batch_idx, (features_b, support_b, label_b, mask_b) in enumerate(train_loader):

            features_b = features_b[0].to(device)
            support_b = support_b[0].to(device)
            label_b = label_b[0].to(device)
            mask_b = mask_b[0].to(device)

            pred = model(support_b, features_b)
            loss = loss_f(pred[mask_b], label_b[mask_b])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*features_b.shape[0]
            train_nodes += features_b.shape[0]

        #=======val======
        val_loss, micro, macro = eval(val_loader)
        print(epoch, train_loss/train_nodes, val_loss, micro, macro)


    #======test=====
    total_loss, micro, macro = eval(test_loader)
    print(total_loss, micro, macro)




if __name__ == '__main__':
    train()
