from utils import load_data
import torch
import argparse
import numpy as np
import torch.optim as optim
import random
import pickle
import torch.nn as nn
import sklearn
from modularity import greedy_modularity_communities, partition, baseline_spectral, make_modularity_matrix
from utils import make_normalized_adj, edge_dropout, negative_sample
from models import GCNLink, GCNClusterNet, GCNDeep, GCNDeepSigmoid, GCN, cluster, GAP
from loss_functions import loss_kcenter, loss_modularity, loss_cut
import copy
from kcenter import CenterObjective, make_all_dists, gonzalez_kcenter, greedy_kcenter, make_dists_igraph, rounding
import networkx as nx
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Enables CUDA training.')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--embed_dim', type=int, default=50,
                    help='Dimensionality of node embeddings')
parser.add_argument('--K', type=int, default=5,
                    help='How many partitions')
parser.add_argument('--negsamplerate', type=int, default=1,
                    help='How many negative examples to include per positive in link prediction training')
parser.add_argument('--edge_dropout', type=float, default=0.2,
                    help='Rate at which to remove edges in link prediction training')
parser.add_argument('--objective', type=str, default='modularity',
                    help='What objective to optimize (currently partitioning or modularity')
parser.add_argument('--dataset', type=str, default='synthetic_spa',
                    help='which network to load')
parser.add_argument('--clustertemp', type=float, default=20,
                    help='how hard to make the softmax for the cluster assignments')
parser.add_argument('--kcentertemp', type=float, default=100,
                    help='how hard to make seed selection softmax assignment')
parser.add_argument('--kcentermintemp', type=float, default=0,
                    help='how hard to make the min over nodes in kcenter training objective')
parser.add_argument('--use_igraph', action='store_true', default=True, help='use igraph to compute shortest paths in twostage kcenter')
parser.add_argument('--train_epochs', type=int, default=5,
                    help='number of training epochs')
parser.add_argument('--num_cluster_iter', type=int, default=1,
                    help='number of iterations for clustering')
parser.add_argument('--singletrain', action='store_true', default=False, help='only train on a single instance')
parser.add_argument('--pure_opt', action='store_true', default=False, help='do only optimization, no link prediction needed')
parser.add_argument('--input_adj_path', type=str)
parser.add_argument('--input_feat_path', type=str)
parser.add_argument('--output_dir', type=str)


args = parser.parse_args()
args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pure_optimization = args.pure_opt

if "tumor" in args.dataset:
    num_graphs = 140
    numtest = 140
if 'diabetes' in args.dataset:
    num_graphs = 681
    numtest = 681
if 'hlt' in args.dataset:
    num_graphs = 4
    numtest = 4

bin_adj_all = []
adj_all = []
adj_train = []
bin_adj_train = []
features_train = []
features_all = []
dist_all = []
dist_train = []

graphs = pickle.load(open(args.input_adj_path, 'rb'))
feats = pickle.load(open(args.input_feat_path, 'rb'))
for sample in graphs:
    for image in graphs[sample]:
        adj_i = nx.adjacency_matrix(graphs[sample][image]) # .toarray()
        adj_i += np.eye(adj_i.shape[0])
        adj_i /= adj_i.sum(axis=1, keepdims=True)
        adj_i = torch.from_numpy(adj_i).float().to_sparse()
        features_i = torch.from_numpy(feats[sample][image])
        if args.objective == 'modularity':
            bin_adj_i = (adj_i.to_dense() > 0).float()
        if args.objective == 'cut':
            bin_adj_i = (torch.from_numpy(nx.adjacency_matrix(graphs[sample][image]).toarray()) > 0).float()
        bin_adj_all.append(bin_adj_i)
        adj_all.append(adj_i.coalesce())
        features_all.append(features_i)

vals = {}
algs = ['ClusterNet']
if args.objective == 'modularity' or args.objective == 'cut':
    ts_algos = ['agglomerative', 'recursive', 'spectral']
elif args.objective == 'kcenter':
    ts_algos = ['gonzalez', 'greedy']
for algo in ts_algos:
    algs.append('train-' + algo)
    algs.append('ts-' + algo)
    algs.append('ts-ft-' + algo)
    algs.append('ts-ft-only-' + algo)
for algo in algs:
    vals[algo] = np.zeros(numtest)

aucs_algs = ['ts', 'ts-ft', 'ts-ft-only']
aucs = {}
for algo in aucs_algs:
    aucs[algo] = np.zeros(numtest)

if args.objective == 'modularity' or args.objective == 'cut':
    mods_test = [make_modularity_matrix(A) for A in bin_adj_all]
    mods_train = [make_modularity_matrix(A) for A in bin_adj_train]
    test_object = mods_test
    train_object = mods_train
    loss_fn = loss_modularity if args.objective == 'modularity' else loss_cut

if pure_optimization:
    train_object = test_object
    adj_train = adj_all
    bin_adj_train = bin_adj_all
    features_train = features_all
    dist_train = dist_all

for test_idx in range(1):
    train_instances = [x for x in range(num_graphs)]
    test_instances = [x for x in range(num_graphs)]
        
    nfeat = features_all[0].shape[1]
    
    K = args.K
    
    if args.objective == 'modularity':
        model_cluster = GCNClusterNet(nfeat=nfeat,
                    nhid=args.hidden,
                    nout=args.embed_dim,
                    dropout=args.dropout,
                    K = args.K, 
                    cluster_temp = args.clustertemp).to(args.device)
    elif args.objective == 'cut':
        model_cluster = GAP(nfeat=nfeat, nhid=args.hidden, nout=args.K, dropout=args.dropout).to(args.device)
    
    #keep a couple of initializations here so that the random seeding lines up
    #with results reported in the paper -- removing these is essentially equivalent to 
    #changing the seed
    _ = GCN(nfeat, args.hidden, args.embed_dim, args.dropout)
    _ = nn.Parameter(torch.rand(K, args.embed_dim))

#    
    # if args.objective == 'modularity':
    #     model_gcn = GCNDeep(nfeat=nfeat,
    #                 nhid=args.hidden,
    #                 nout=args.K,
    #                 dropout=args.dropout, 
    #                 nlayers=2)
    # if args.objective == 'cut':
    #     model_gcn = GAP(nfeat=nfeat, nhid=args.hidden, nout=args.K, dropout=args.dropout)
    
    optimizer = optim.Adam(model_cluster.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    
    losses = []
    losses_test = []
    num_cluster_iter = args.num_cluster_iter
        
    def get_average_loss(model, adj, bin_adj_for_loss, objectives, instances, features, num_reps = 10, hardmax = False, update = False, algoname =  None):
        if hardmax:
            model.eval()
        loss = 0
        for _ in range(num_reps):
            for idx, i in enumerate(instances):
                features_i = features[i].to(args.device)
                adj_i = adj[i].to(args.device)
                bin_adj_for_loss_i = bin_adj_for_loss[i].to(args.device)
                objectives_i = objectives[i].to(args.device)
                if args.objective == 'modularity':
                    mu, r, embeds, dist = model(features_i, adj_i, num_cluster_iter)
                    if hardmax:
                        r = torch.softmax(100*r, dim=1)
                    this_loss = -loss_fn(mu, r, embeds, dist, bin_adj_for_loss_i, objectives_i, args)
                elif args.objective == 'cut':
                    Y = model(features_i, bin_adj_for_loss_i)
                    this_loss = loss_fn(Y, bin_adj_for_loss_i)
                loss += this_loss.cpu()
                if update:
                    vals[algoname][test_instances.index(i)] = this_loss.item()
        if hardmax:
            model.train()
        return loss/(len(instances)*num_reps)

    #Decision-focused training
    if True:
        min_loss_test, min_e, min_model_cluster = np.inf, 0, None
        for e in range(args.train_epochs):
            model_cluster.train()
            for t, i in enumerate(sklearn.utils.shuffle(train_instances)):
                features_train_i = features_train[i].to(args.device)
                adj_train_i = adj_train[i].to(args.device)
                bin_adj_train_i = bin_adj_train[i].to(args.device) 
                train_object_i = train_object[i].to(args.device)
                if args.objective == 'modularity':
                    mu, r, embeds, dist = model_cluster(features_train_i, adj_train_i, num_cluster_iter)
                    loss = loss_fn(mu, r, embeds, dist, bin_adj_train_i, train_object_i, args)
                if args.objective == 'cut':
                    Y = model_cluster(features_train_i, bin_adj_train_i)
                    loss = loss_fn(Y, bin_adj_train_i)
                if args.objective != 'kcenter' and args.objective != 'cut':
                    loss = -loss
                optimizer.zero_grad()
                loss.backward()
                if t % 100 == 0 and t != 0:
                    num_cluster_iter = 5
                # if t % 10 == 0:
                #     if args.objective == 'modularity':
                #         r = torch.softmax(100*r, dim=1)
                    ## loss_train = get_average_loss(model_cluster, adj_train, bin_adj_train, bin_adj_all, test_object, train_instances, features_train, hardmax=True)
                    # loss_test = get_average_loss(model_cluster, adj_train, bin_adj_train, bin_adj_all, test_object, test_instances, features_train, hardmax=True)
                    ## loss_valid = get_average_loss(model_cluster, adj_train, bin_adj_train, bin_adj_all, test_object, valid_instances, features_train, hardmax=True)
                    # losses_test.append(loss_test.item())
                    # print(t, loss_train.item(), loss_test.item(), loss_valid.item())
                    # print(e, t, loss_test.item())
                # losses.append(loss.item())
                optimizer.step()
            model_cluster.eval()
            with torch.no_grad():
                if args.objective == 'modularity' or args.objective == 'cut':
                    loss_test = get_average_loss(model_cluster, adj_train, bin_adj_train, train_object, train_instances, features_train, hardmax=True)
                if loss_test < min_loss_test:
                    min_loss_test, min_e, min_model_cluster = loss_test, e, copy.deepcopy(model_cluster)
                print(e, loss_test, min_e, min_loss_test)
                if e - min_e >= 3 or e == args.train_epochs - 1:
                    if args.objective == 'modularity':
                        r_sep = []
                        all_embeds = []
                        for i in train_instances:
                            mu, r, embeds, dist = min_model_cluster(features_train_i, adj_train_i, num_cluster_iter)
                            all_embeds.append(embeds.cpu())
                            r_sep.append(r.cpu())
                        r_sep = torch.vstack(r_sep)
                        torch.save(r_sep, open(f'{args.output_dir}/first_{e}_{args.clustertemp}_sep', 'wb'))
                        all_embeds = torch.cat(all_embeds, dim=0)
                        torch.save(r, open(f'{args.output_dir}/first_emb_{e}_{args.clustertemp}', 'wb'))
                        mu_init, _, _ = cluster(all_embeds, args.K, 1, num_cluster_iter, cluster_temp = args.clustertemp, init = min_model_cluster.init)
                        mu, r, dist = cluster(all_embeds, args.K, 1, 1, cluster_temp = args.clustertemp, init = mu_init.detach().clone())
                        torch.save(r.cpu(), open(f'{args.output_dir}/first_{e}_{args.clustertemp}', 'wb'))
                        torch.save(min_model_cluster.state_dict().cpu(), open(f'{args.output_dir}/first_model_{e}_{args.clustertemp}', 'wb'))
                        pickle.dump(min_loss_test, open(f'{args.output_dir}/results_distributional_{e}_{args.clustertemp}.pickle', 'wb'))
                    elif args.objective == 'cut':
                        Ys, labels = [], []
                        for i in train_instances:
                            Y = model_cluster(features_train_i, bin_adj_train_i)
                            Ys.append(Y)
                            labels.append(Y.max(dim=1))
                        torch.save(Ys.cpu(), open('Ys', 'wb'))
                        torch.save(labels.cpu(), open('labels', 'wb'))
                    break
        # if args.objective == 'modularity':
        #     loss_test = get_average_loss(model_cluster, adj_train, bin_adj_train, bin_adj_all, test_object, test_instances, features_train, hardmax=True, update = True, algoname = 'ClusterNet')
        # print('after training', np.mean(vals['ClusterNet'][:numtest]), np.std(vals['ClusterNet']))
    