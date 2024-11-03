import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree, add_self_loops

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA

from backbone import *

class Ours(nn.Module):
    '''
    hpyer paramter:
    dim :
    alpha :
    '''
    def __init__(self, d, c, args):
        super(Ours, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout, use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'h2gcn':
            self.encoder = H2GCN(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device, logit=True):
        '''return predicted logits'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        if(logit):
            return self.encoder(x, edge_index)
        else:
            return self.encoder.intermediate_forward(x, edge_index)

    @torch.no_grad()
    def detect(self, dataset_train, tarin_idx, dataset_test, test_idx, device, args):
        '''return ood score compute by feature'''
        logits_test = self.forward(dataset_test, device)
        feature_train = self.forward(dataset_train, device, False)[tarin_idx]
        feature_test = self.forward(dataset_test, device, False)
        max_logit = logits_test.max(dim=1)[0]

        _, _, v = torch.pca_lowrank(feature_train, q=min(feature_train.shape[0], feature_train.shape[1]))
        std_feature = (feature_test - torch.mean(feature_train, dim=0)) / torch.std(feature_train, dim=0, correction=0)
        test_projection = torch.matmul(feature_test, v[:, :args.nc_dim])
        norm_all = torch.linalg.vector_norm(feature_test, dim = 1)
        norm_p = torch.linalg.vector_norm(test_projection, dim = 1)
        score = norm_p #/ norm_all
        if args.scale:
            score *= max_logit
        
        if args.use_prop:
            edge_index = dataset_test.edge_index.to(device)
            score = self.propagation(score, edge_index, feature_test, prop_layers=args.K, alpha=args.alpha)

        return score[test_idx].cpu()
    
    def propagation(self, score, edge_index, feature_test, prop_layers=1, alpha=0.5):
        score = score.unsqueeze(1)
        node_num = score.shape[0]
        add_self_loops(edge_index)
        row, col = edge_index
        edge_weight = F.cosine_similarity(feature_test[row], feature_test[col], dim=1)
        deg = torch.zeros(node_num, dtype=torch.float).to('cuda')
        deg.scatter_add_(0, row, edge_weight)
        deg[torch.nonzero(deg == 0)] = 1
        norm_edge_weight = edge_weight / deg[row]
        adj = SparseTensor(row=row, col=col, value=norm_edge_weight, sparse_sizes=(node_num, node_num))
        for _ in range(prop_layers):
            score = score * alpha + matmul(adj, score) * (1 - alpha)
        score = torch.nan_to_num(score, nan=0.)
        return score.squeeze(1)

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        edge_index_out, edge_index_in = dataset_ood.edge_index.to(device), dataset_ind.edge_index.to(device)

        # get predicted logits from gnn classifier
        logits_in = self.forward(dataset_ind, device)
        logits_out = self.forward(dataset_ood, device)

        train_in_idx, train_ood_idx = dataset_ind.splits['train'],  dataset_ood.node_idx

        # compute supervised training loss
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        loss = sup_loss

        return loss
