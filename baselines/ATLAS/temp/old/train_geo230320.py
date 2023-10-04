from typing import Callable, Dict, List, Optional, Tuple, Union
import random
import torch
import time
import sys
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import from_networkx
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import networkx as nx

from read_ALTAS_graph import get_ATLAS_G

device =  torch.device('cpu') # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
steps = 200
lr = 0.002
batch_size = 512
thres = 1.5
NORMALIZE = False

class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, concat=False):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 32, normalize=False, concat=concat)
        self.conv2 = SAGEConv(32, out_channels, normalize=False, concat=concat)

    def forward(self, x, edge_index):    
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=1)
    
        # data = data_flow[0]
        # x = x[data.n_id]
        # x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        # x = F.dropout(x, p=0.5, training=self.training)
        # data = data_flow[1]
        # x = self.conv2((x, None), data.edge_index, size=data.size)
        # return F.log_softmax(x, dim=1)


## get data for training GNN
def get_data_from_nx(G:nx.MultiDiGraph):
    
    elabel2no, elabel_cnt = {}, 0 # edge_type to no. map
    for h, t, r in G.edges(data=True):
        if r['type'] not in elabel2no:
            elabel2no[r['type']] = elabel_cnt
            elabel_cnt += 1

    print('elabel_cnt',elabel_cnt)
    # print(elabel2no)

    nid2no, no2nid = {},{} # node_id to no. map
    nlabel2no, nlabel_cnt = {}, 0 # node_type to no. map

    n_cnt = len(G.nodes())
    X, y = np.zeros((n_cnt,elabel_cnt*2)), [] # GNN feat. and label
    for i, (nid, nattr) in enumerate(G.nodes(data=True)):
        nid2no[nid] = i
        no2nid[i] = nid
        if nattr['type'] not in nlabel2no:
            nlabel2no[nattr['type']] = nlabel_cnt
            nlabel_cnt += 1
        y.append(nlabel2no[nattr['type']])
        
        for _,_,r in G.in_edges(nid, data=True):
            X[i][elabel2no[r['type']]] += 1
        for _,_,r in G.out_edges(nid, data=True):
            X[i][elabel2no[r['type']]+elabel_cnt] += 1
            
    print('nlabel_cnt', nlabel_cnt)
    # print('nlabel2no', nlabel2no)
    # print('X.shape', X.shape)
    # print('len(y)',len(y))
    
    return X, y, nid2no, no2nid # nid2no, nlabel2no, elabel2no, elabel_cnt, nlabel_cnt

 
## get mal node id from keyword of ALTAS
def get_mal_node_id(G:nx.MultiDiGraph,
                    mal_keyword:List[str]
                    ):
    
    mal_node_cnt = 0
    mal_nid = []
    
    for i, label in enumerate(G.nodes()):
        for mal in mal_keyword:
            if mal in label:
                mal_nid.append(label)
                mal_node_cnt += 1
                break
    print('mal_node_num:', mal_node_cnt)
    return mal_nid

## 0-1 normalize of torch.Tensor
def normalize_tensor(t:torch.Tensor):
    # get the minimum and maximum values along dimension 0
    min_val = t.min(dim=0)[0].unsqueeze(0)
    max_val = t.max(dim=0)[0].unsqueeze(0)

    # subtract the minimum from each element and divide by the range
    normalized_t = (t - min_val) / (max_val - min_val)

    return normalized_t
    
## get training data of GNN
def Graph2trainset(G:nx.MultiDiGraph,
               mal_keyword:List[str],
               ) -> (NeighborLoader):
    
    print('\nProcess Graph for training:', G.name)
    
    # get format data from Graph    
    X, y, nid2no, no2nid = get_data_from_nx(G)
    
    benign_mask = np.asarray([True]*X.shape[0])
    mal_nid = get_mal_node_id(G, mal_keyword)
    for id in mal_nid:
        benign_mask[nid2no[id]] = False
    print('Benign node num:', benign_mask.sum())
    
    train_mask = np.asarray([False]*X.shape[0])
    train_ratio = 1.0 ## only used for single graph train and evluation
    ben_idx = np.where(benign_mask)[0]
    select_train_idx = random.sample(list(ben_idx), int(len(ben_idx)*train_ratio))
    train_mask[select_train_idx] = True
    print('Train node num:', train_mask.sum())
    
    select_nids = list(map(no2nid.get, select_train_idx))
    sub_ben_G = G.subgraph(select_nids)
    
    X_train, y_train, nid2no, no2nid = get_data_from_nx(sub_ben_G)
    # fmt_G = nx.DiGraph(sub_ben_G)
    fmt_G = nx.relabel_nodes(sub_ben_G,nid2no)
    
    # get training subgraph edges
    train_edges = fmt_G.edges()
    train_edge_s = np.asarray(list(train_edges))[:,0]
    train_edge_e = np.asarray(list(train_edges))[:,1]
    
    # get training data of GNN
    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    if NORMALIZE: X_train = normalize_tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    train_edge_index = torch.tensor(np.asarray([train_edge_s, train_edge_e]), dtype=torch.long).to(device)
    train_data = Data(x=X_train, y=y_train, edge_index=train_edge_index, benign_mask=benign_mask)
    
    train_loader = NeighborLoader(train_data, num_neighbors=[-1, -1], 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  )
    
    return train_loader
    
## get test data of GNN
def Graph2testset(G:nx.MultiDiGraph,
               mal_keyword:List[str],
               ) -> (Data):
    
    print('\nProcess Graph for testing:', G.name)
    
    # get format data from Graph    
    X, y, nid2no, no2nid = get_data_from_nx(G)
    
    # format MultiDiGraph to DiGraph and relabel node id with No.
    # fmt_G = nx.DiGraph(G)
    fmt_G = nx.relabel_nodes(G,nid2no)
    
    edges = fmt_G.edges()
    edge_s = np.asarray(list(edges))[:,0]
    edge_e = np.asarray(list(edges))[:,1]
    
    # get masks
    benign_mask = np.asarray([True]*X.shape[0])
    mal_nid = get_mal_node_id(G, mal_keyword) # mal nids in which nids is str
    mal_nidno = [] # mal nids in which nids is no.
    for id in mal_nid:
        benign_mask[nid2no[id]] = False
        mal_nidno.append(nid2no[id])
        # X[nid2no[id]][0] = 1000000
    print('Benign node num:', benign_mask.sum())
    
    # to tensor, generate Data Loader
    X = torch.tensor(X, dtype=torch.float).to(device)
    if NORMALIZE: X = normalize_tensor(X)
    y = torch.tensor(y, dtype=torch.long).to(device)
    edge_index = torch.tensor(np.asarray([edge_s, edge_e]), dtype=torch.long).to(device)
    
    test_data = Data(x=X, y=y,edge_index=edge_index,benign_mask=benign_mask) 
    
    return test_data, fmt_G, mal_nidno


def get_2hop_neighbors(G, node):
    two_hop_neighbors = set()
    for n in G.neighbors(node):
        two_hop_neighbors.update(G.neighbors(n))
    return two_hop_neighbors

# evaluation for threatrace
def eval_metric(
        y_pred:torch.Tensor, 
        y_true:torch.Tensor, 
        benign_mask:np.ndarray,
    ):
    
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    # benign_mask = benign_mask.cpu().numpy()

    TN = np.sum((y_pred == y_true) * benign_mask)
    TP = np.sum((y_pred != y_true) * (1-benign_mask))
    FP = np.sum(benign_mask) - TN
    FN = np.sum((1-benign_mask)) - TP
    
    total = TP+TN+FP+FN
    
    ben_class_dist = [] 
    pred = y_pred[benign_mask]
    for i in range(7):
        ben_class_dist.append(len(pred[pred==i]))
    
    mal_class_dist = [] 
    pred = y_pred[np.asarray(1-benign_mask).astype(np.bool)]
    for i in range(7):
       mal_class_dist.append(len(pred[pred==i]))
            
    print(f'(ben_class_dist: {ben_class_dist}, mal_class_dist: {mal_class_dist}')
    
    return f'TP:{TP:>3}, TN:{TN:>6}, FN:{FN:>3}, FP:{FP:>6} (Total:{total:>6})'

# evaluation for threatrace: consider logits.max()/logits.second() > thres
def eval_metric_thres(out:torch.Tensor, 
            y_true:torch.Tensor, 
            benign_mask:np.ndarray,
            ):
    
    out  = F.softmax(out, dim=1)
    y_pred = out.argmax(dim=1) # type of node, not mal/ben label
    
    # max_logits = out.max()
    # snd_logits = out.max()
    
    confidence_mask = [] # whether have enough confidence (>thres), if not, get mal label 
    top2 = torch.topk(out, 2)[0] 
    ratio = top2[:, 0]/top2[:, 1]
    for r in ratio:
        if (r>1. and r>=thres) or (r<1. and r<=1/thres):
            confidence_mask.append(1)
        else:
            confidence_mask.append(0)

    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    confidence_mask = np.asarray(confidence_mask)
    
    TN = np.sum((y_pred == y_true) * confidence_mask * benign_mask)
    TP = np.sum((y_pred != y_true) * (1-benign_mask)) + np.sum((y_pred == y_true) * (1-confidence_mask) * (1-benign_mask))
    FP = np.sum(benign_mask) - TN
    FN = np.sum((1-benign_mask)) - TP
    total = TP+TN+FP+FN
    
    return f'TP:{TP:>3}, TN:{TN:>6}, FN:{FN:>3}, FP:{FP:>6} (Total:{total:>6})'

# evaluation for threatrace: consider 2-hop analysis to reduce FP and FN
# evaluation for threatrace: consider top-2 logits both to reduce FP (like Deeplog, if top-2_pred_logits==y_true, then raise a Neg)
def eval_metric_2hop_top2(
        out:torch.Tensor, 
        y_true:torch.Tensor, 
        benign_mask:np.ndarray,
        test_G:nx.DiGraph,
        mal_nidno:List[int],
    ):
    
    out  = F.softmax(out, dim=1)
    top2_indices = out.topk(2, dim=1)[1]
    
    y_pred = out.argmax(dim=1) # type of node, not mal/ben label
    
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    TN = np.sum((y_pred == y_true) * benign_mask)
    TP = np.sum((y_pred != y_true) * (1-benign_mask))
    
    strict_FP_nid = np.where(((y_pred!=y_true)*benign_mask)==1)[0]
    hop_TN_nid = [] # if node in strict_FP_nid can reach MAL (ground-truth) neighbor in 2-hop, it will be in hop_TN_nid
    top_TN_nid = []
    
    for nid in strict_FP_nid:
        nb_2hops = get_2hop_neighbors(test_G, nid)
        for no in mal_nidno:
            if no in nb_2hops:
                hop_TN_nid.append(nid)
                TN += 1
                break
        if nid not in hop_TN_nid:
            if y_true[nid] in top2_indices[nid]:
                TN += 1
                top_TN_nid.append(nid)
                        
    strict_FN_nid = np.where(((y_pred==y_true)*(1-benign_mask))==1)[0]
    
    # in top-2 eval, y_pred != y_true is not enough to describe Pos but y_pred_top2 != y_true, so we need to remove y_pred_2nd == y_true as a Neg
    for id in np.where(benign_mask==0): 
        if y_true[id] == top2_indices[id][1]:
            TP -= 1
            strict_FN_nid = np.append(strict_FN_nid, id)
    
    hop_TP_nid = [] # if node in strict_FN_nid can be reached by Pos (predicted) neighbor in 2-hop, it will be in hop_TP_nid
    
    rev_test_G = test_G.reverse()
    for nid in strict_FN_nid:
        nb_2hops = get_2hop_neighbors(rev_test_G, nid)
        for no in nb_2hops:
            if y_pred[no] != y_true[no]:
                hop_TP_nid.append(nid)
                TP += 1
                break
            
    FP = np.sum(benign_mask) - TN
    FN = np.sum((1-benign_mask)) - TP
    total = TP+TN+FP+FN
    
    print(f' (hop_TN_nid:{len(hop_TN_nid)}, top_TN_nid:{len(top_TN_nid)}, hop_TP_nid:{len(hop_TP_nid)})')
    return f'TP:{TP:>3}, TN:{TN:>6}, FN:{FN:>3}, FP:{FP:>6} (Total:{total:>6})'
    
def train_single_model(
    train_loaders:List[NeighborLoader],
    test_data:Data,
    # vali_data:Data,
    test_G:nx.DiGraph,
    mal_nidno:List[int],
    model_pth:str, 
    ):

    feature_num = test_data.x.shape[1]
    label_num = (max(test_data.y)+1).item()
    model = SAGENet(feature_num, label_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) 
    
    # begin training
    print('\nBegin training single-model threatrace')
    for epoch in range(steps):
        total_loss = 0

        model.train()
        for train_loader in train_loaders:
            for batch in iter(train_loader):
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss
        
        if epoch % 10 == 0: 
            model.eval()
            out = model(test_data.x, test_data.edge_index)
            # vout = model(vali_data.x, vali_data.edge_index)
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss/len(train_loader):.3f}', 
                    # '| validation:', eval_metric(vout, vali_data.y, vali_data.benign_mask),
                    '| test:', eval_metric(out.argmax(dim=1), test_data.y, test_data.benign_mask),
                    '| test_hop:',eval_metric_2hop_top2(out, test_data.y, test_data.benign_mask, test_G, mal_nidno)
                  )
    exit(-1)
    torch.save(model.state_dict(), model_pth)


if __name__ == "__main__":
    
    folder = 'ATLAS/paper_experiments/S1/output'
    
    S1_log_file = 'testing_preprocessed_logs_S1-CVE-2015-5122_windows'
    S1_mal = ['0xalsaheel.com', '192.168.223.3', 'payload.exe']
    
    S2_log_file = 'training_preprocessed_logs_S2-CVE-2015-3105_windows'
    S2_mal = ['0xalsaheel.com', '192.168.223.3', 'payload.exe']

    S3_log_file = 'training_preprocessed_logs_S3-CVE-2017-11882_windows'
    S3_mal = ['0xalsaheel.com', '192.168.223.3', 'payload.exe', 'msf.rtf', 'msf.exe', 'aalsahee/index.html']
    
    S4_log_file = 'training_preprocessed_logs_S4-CVE-2017-0199_windows_py'
    S4_mal = ['0xalsaheel.com', '192.168.223.3', 'msf.doc', 'pypayload.exe', 'aalsahee/index.html']

    G1 = get_ATLAS_G(folder+S1_log_file)
    G2 = get_ATLAS_G(folder+S2_log_file)
    G3 = get_ATLAS_G(folder+S3_log_file)
    G4 = get_ATLAS_G(folder+S4_log_file)
    
    train_loaders = []
    train_loaders.append(Graph2trainset(G1,S1_mal))
    train_loaders.append(Graph2trainset(G2,S2_mal))
    train_loaders.append(Graph2trainset(G3,S3_mal))
    train_loaders.append(Graph2trainset(G4,S4_mal))
    
    test_data, fmt_G, mal_nidno = Graph2testset(G1,S1_mal)
    # test_data, fmt_G, mal_nidno = Graph2testset(G2,S2_mal)
    # test_data, fmt_G, mal_nidno = Graph2testset(G3,S3_mal)
    # test_data, fmt_G, mal_nidno = Graph2testset(G4,S4_mal)
    
    # vali_data = Graph2testset(G2,S2_mal)
    
    model_pth = 'save/threatrace_train_s1s2s3s4_test_s1.pt'
    
    train_single_model(train_loaders, test_data, nx.DiGraph(fmt_G), mal_nidno, model_pth)