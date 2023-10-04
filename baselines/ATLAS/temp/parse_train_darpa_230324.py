# parse darpa graph 
from typing import Callable, Dict, List, Optional, Tuple, Union
import random
import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import _pickle as pkl
import dgl
from dgl.nn import SAGEConv

device = torch.device('cpu')
steps = 300
lr = 0.0001
batch_size = 2048
hidden_layer = 32

MAX_TEST_BATCH = 2000000

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type="mean")
        self.h_feats = h_feats

    def forward(self, mfgs, x, edge_weights):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[: mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst), edge_weight=edge_weights[0])  # <---
        h = F.relu(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]  # <---
        # h_dst = h[: len(mfgs[-1].dstdata['feat'])] 
        # print('h_dst', h_dst.shape, mfgs[1].num_dst_nodes())
        # print(len(mfgs[-1].dstdata['feat']), h.shape)
        h = self.conv2(mfgs[1], (h, h_dst), edge_weight=edge_weights[1])  # <---
        return F.log_softmax(h, dim=1)
    
# parse Threatrace parse_darpa output.txt into nx.MultiDiGraph 
# str(srcId) + '\t' + str(srcType) + '\t' + str(dstId) + '\t' + str(dstType) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
def parse_graph_darpa(
        threatrace_parse_pth:str,   # threatrace parse_darpa output.txt file path
        save_pth_prefix:str,        # files name prefix to save
        meta = None, # whether extract from existing meta 
        verbose = True,
    )->(nx.MultiDiGraph):
    
    if meta is None:
        elabel2no, no2elabel, elabel_cnt = {}, {}, 0 # edge_type to no. map
        nlabel2no, no2nlabel, nlabel_cnt = {}, {}, 0 # node_type to no. map
    else:
        elabel2no, no2elabel, elabel_cnt = meta['elabel2no'], meta['no2elabel'], meta['elabel_cnt']
        nlabel2no, no2nlabel, nlabel_cnt = meta['nlabel2no'], meta['no2nlabel'], meta['nlabel_cnt']
        
    nid2no, no2nid = {},{} # node_id to no. map
    G = nx.MultiDiGraph()
    
    with open(threatrace_parse_pth, "r") as f:
        for i, line in enumerate(f):
            arr = line.strip().split()
            if verbose and i % 1e6 == 0: print(f'(Info: parse darpa graph for threatrace {int(i/1e6)} (M))') 
            G.add_edge(arr[0], arr[2], type=arr[4], timestamp=arr[5])
            G.add_node(arr[0], type=arr[1])
            G.add_node(arr[2], type=arr[3])
            if (meta is None) and (arr[4] not in elabel2no):
                elabel2no[arr[4]] = elabel_cnt
                no2elabel[elabel_cnt] = arr[4]
                elabel_cnt += 1

    pkl.dump(G, open(f'save/{save_pth_prefix}_nxG.pkl', 'wb'))
    if verbose: print(f'(Info: successfully save nx.MultiDiGraph at {save_pth_prefix}_nxG.pkl')

    n_cnt = len(G.nodes())
    # print(n_cnt, elabel_cnt)
    X, y = np.zeros((n_cnt,elabel_cnt*2)), [] # GNN feat. and label
    for i, (nid, nattr) in enumerate(G.nodes(data=True)):
        if verbose and i%1e5 == 0: print(f'(Info: extract node features and labels for threatrace {int(i/1e5)} (10W))')
        nid2no[nid] = i
        no2nid[i] = nid
        if (meta is None) and (nattr['type'] not in nlabel2no):
            nlabel2no[nattr['type']] = nlabel_cnt
            no2nlabel[nlabel_cnt] = nattr['type']
            nlabel_cnt += 1
        y.append(nlabel2no[nattr['type']])
        
        for _,_,r in G.in_edges(nid, data=True):
            X[i][elabel2no[r['type']]] += 1
        for _,_,r in G.out_edges(nid, data=True):
            X[i][elabel2no[r['type']]+elabel_cnt] += 1
    
    print('X.shape', X.shape)
    np.savez_compressed(f'save/{save_pth_prefix}_data.npz', X=X, y=y)
    if verbose: print(f'(Info: successfully save node feat/label at {save_pth_prefix}_data.npz')
    
    if meta is None:
        meta = {} # meta info (e.g. maps) of graph
        meta['elabel2no'] = elabel2no
        meta['no2elabel'] = no2elabel
        meta['nlabel2no'] = nlabel2no
        meta['no2nlabel'] = no2nlabel
        meta['elabel_cnt'] = elabel_cnt
        meta['nlabel_cnt'] = nlabel_cnt
    else:
        print('NOTICE: parse with existing meta (feature/lable maps)!')

    meta['nid2no'] = nid2no
    meta['no2nid'] = no2nid
    pkl.dump(meta, open(f'save/{save_pth_prefix}_meta.pkl', 'wb'))
    if verbose: print(f'(Info: successfully save meta infos at {save_pth_prefix}_meta.pkl')
        
        
# get malicious nid from G and all_attack_nid (from threatrace)
def get_mal_nid_darpa(G:nx.MultiDiGraph,
                    attack_nid_pth:str, # all_attack_nid file pth (from threatrace)
                    nid2no:dict,
                    ):
    print('NOTICE: Get malicious nid in current graph from all attack node ids')
    # mal_node_cnt = 0
    mal_nids, mal_nidnos = [],[]
    
    all_mal_nids = []
    with open(attack_nid_pth, "r") as f:
        for i, line in enumerate(f):
            all_mal_nids.append(line.strip())
    
    # nids = list(G.nodes())
    # for mid in all_mal_nids:
    #     if mid in nids:
    #         mal_nids.append(mid)
    #         mal_node_cnt += 1
    print('all_mal_nids:', len(all_mal_nids))
    for mid in all_mal_nids:
        if mid in nid2no:
            mal_nids.append(mid)
            mal_nidnos.append(nid2no[mid])

    # print('mal_node_num:', mal_node_cnt)
    print('mal_node_num:', len(mal_nids))
    
    ### 注意：要去重！！！！ 不然DGL MGFS 会报错，详见 https://github.com/dmlc/dgl/issues/4512
    mal_nids = list(set(mal_nids))
    mal_nidnos = list(set(mal_nidnos))
    print('reduce duplicate elements:', len(mal_nids))
    
    # mal_nidnos = list(map(nid2no.get, mal_nids))
    return mal_nids, mal_nidnos

# Get train-/test-able DGLGraph from RAW nxGraph
def nx2dglGraph_darpa(
        G:nx.MultiDiGraph,
        meta:dict, 
        node_data:dict,
        mal_nid:List[str],
    ) -> (dgl.DGLGraph):
    
    print('NOTICE: Transfer nx.MultiDiGraph to DGLGraph')
     
    nid2no = meta['nid2no']
    elabel2no = meta['elabel2no']
    
    s_edges, e_edges = [], []
    edge_weights = []
    edge_type_no = []
    H = nx.DiGraph(G)
    for u, v, attr in H.edges(data=True):
        s_edges.append(nid2no[u])
        e_edges.append(nid2no[v])
        edge_count = G.number_of_edges(u, v)
        edge_weights.append(edge_count)
        edge_type_no.append(elabel2no[attr['type']])
            
    dg = dgl.graph((s_edges, e_edges))
    dg.ndata['feat'] = torch.tensor(node_data['X'], dtype=torch.float)
    dg.ndata['label'] = torch.tensor(node_data['y'], dtype=torch.long)
    dg.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float)
    dg.edata['etype'] = torch.tensor(edge_type_no, dtype=torch.long)
    
    benign_mask = np.asarray([True]*len(node_data['y']))
    for id in mal_nid:
        benign_mask[nid2no[id]] = False
    print('benign node num:', benign_mask.sum(),'all node num:', len(benign_mask))
    
    dg.ndata['ben_mask'] = torch.tensor(benign_mask, dtype=torch.bool)
          
    return dg 

# Get train-/test dataLoader from DGLGraph
def get_dglLoder_darpa(
        dg:dgl.DGLGraph,
        is_train = True,
    ):
    print(f'+++++++++++++++++++++++++++ Threatrace CONFIG +++++++++++++++++++++++++++\n'
        f'batch size                  : {batch_size}\n'
        f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        
    print(f"NOTICE: Get {'TRAIN' if is_train else 'TEST'} dataLoader from DGLGraph")
    sampler = dgl.dataloading.NeighborSampler([-1, -1])
    if is_train:
        train_dataloader = dgl.dataloading.DataLoader(
            # The following arguments are specific to DGL's DataLoader.
            dg,  # The graph
            dg.nodes()[dg.ndata['ben_mask']],  # The node IDs to iterate over in minibatches
            sampler,  # The neighbor sampler
            device=device,  # Put the sampled MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,  # Batch size
            shuffle=True,  # Whether to shuffle the nodes for every epoch
            drop_last=False,  # Whether to drop the last incomplete batch
            num_workers=0,  # Number of sampler processes
        )
        return train_dataloader
    
    else:
        valid_dataloader = dgl.dataloading.DataLoader(
            dg,
            dg.nodes(),
            sampler,
            batch_size=MAX_TEST_BATCH,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            device=device,
        )
        return valid_dataloader

# get class weight for imbalance dataset. 
# update at 03.23: add mal_class_mask
def get_loss_weight(
        y_train: torch.Tensor,
        num_classes: int,
        min_thres=5,
        mal_class_mask=None, # if not None, mask class that not appear in mal nodes  
    ) -> (torch.Tensor):
    num = []
    weight = []
    y_train = y_train.numpy()
    for i in range(num_classes):
        num.append(len(y_train[y_train==i]))
        if num[i] == 0:
            weight.append(0.)
        elif num[i] <= min_thres: # overlook too small class (darpa)
            if (mal_class_mask is not None):
                if mal_class_mask[i]==0:
                    weight.append(1.)
                    print(f'**WARNING**: class {i} has too small samples {num[i]}<={min_thres} AND NOT APPEAR IN MAL NODES to reweight this class as 1')    
                else:
                    weight.append(len(y_train)/num[i])
            else:
                weight.append(1.)
                print(f'**WARNING**: class {i} has too small samples {num[i]}<={min_thres} reweight this class as 1')    
        else:
            weight.append(len(y_train)/num[i])
    print('Train Each Class Num:', num)
    print('Weight Each Class Num:', weight)
    return torch.tensor(weight, dtype=torch.float).to(device)

# used for generate the params `mal_class_mask` for func get_loss_weight()
def get_mal_class_mask(dg, meta, mal_nidnos):
    mal_label = dg.ndata['label'].numpy()[mal_nidnos]
    mal_class_mask = []
    for i in range(meta['nlabel_cnt']):
        # print(f"class {i} ({meta['no2nlabel'][i]}) mal node number: {len(mal_label[mal_label==i])}")
        mal_class_mask.append(len(mal_label[mal_label==i]))
    return np.asarray(mal_class_mask)
    
def evaluation(
        y_pred:np.ndarray, 
        y_true:np.ndarray, 
        benign_mask:np.ndarray,
    ):

    TN = np.sum((y_pred == y_true) * benign_mask)
    TP = np.sum((y_pred != y_true) * (1-benign_mask))
    FP = np.sum(benign_mask) - TN
    FN = np.sum((1-benign_mask)) - TP
    
    total = TP+TN+FP+FN
    
    return TP, TN, FP, FN

# Train single model of threatrace for darpa dataset
def train_threatrace_darpa(
    train_loaders:List[dgl.dataloading.DataLoader],
    valid_dataloader:dgl.dataloading.DataLoader,
    model_pth:str,
    meta:dict,
    select_model='TPFP', # 'TPFP' save model when TP and FP both become better; 
                         # 'score': weight heuristic score TP/TN weight is Neg/Pos, 
                         # 'score/x': heuristic score /x common is 2 or 3 (lower the weight on TP, its okay if TP reduced a little but FP reduce more)
                         # 'score/x/y': increase x will lower the penaility if TP reduce, y is the minimum TP
    steps=steps,
    lr = lr,
    check_itv = 5, # interval epoches to check whether best model,
    weight_decay = 1e-8,
    is_weighted = False, # if True, activate get_loss_weight()
    mal_class_mask = None, # params for get_loss_weight()
    min_weight_thres = 5, # params for get_loss_weight()
    ):
    
    print(f'+++++++++++++++++++++++++++ Threatrace CONFIG +++++++++++++++++++++++++++\n'
            f'learning rate               : {lr}\n'
            f'training epoches            : {steps}\n'
            f'check best modek interval   : {check_itv}\n'
            f'batch size                  : {batch_size}\n'
            f'hidden_layer size           : {hidden_layer}\n'
            f'weight_decay                : {weight_decay}\n'
            f'select_model                : {select_model}\n'
            f'weighted class?             : {is_weighted}\n')
    if is_weighted:
        print(f' - minimum number thres     : {min_weight_thres}\n'
              f' - mask unappeared in mal?  : {True if (mal_class_mask is not None) else False}\n')
    print(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    num_features, num_classes = meta['elabel_cnt']*2, meta['nlabel_cnt']
    
    model = Model(num_features, hidden_layer, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if select_model == 'TPFP':
        best_tp, best_fp = 0, np.inf # use for save best model
    elif 'score' in select_model:
        best_score = -np.inf
        
    if is_weighted:
        labels = []
        for _, _, mfgs in valid_dataloader:
            labels = mfgs[-1].dstdata["label"]
        class_weight = get_loss_weight(labels, num_classes, min_weight_thres, mal_class_mask)
    
    is_save_model = False
    for epoch in range(steps):
        model.train()
        total_loss = 0
        
        for train_loader in train_loaders:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(train_loader): 
                
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]
                edge_weights = [mfgs[0].edata['weight'], mfgs[1].edata['weight']]
                
                out = model(mfgs, inputs, edge_weights)
                
                if is_weighted:
                    loss = F.nll_loss(out, labels, weight=class_weight)
                else:
                    loss = F.nll_loss(out, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
        
        if (epoch==steps-1) or (epoch % check_itv == 0 and epoch > 0): 
            model.eval()
            predictions = []
            labels = []
            ben_mask = []
            for input_nodes, output_nodes, mfgs in valid_dataloader:
                inputs = mfgs[0].srcdata["feat"]
                edge_weights = [mfgs[0].edata['weight'], mfgs[1].edata['weight']]
                # ben_mask.append(mfgs[0].srcdata["ben_mask"].cpu().numpy())
                ben_mask.append(mfgs[-1].dstdata["ben_mask"].cpu().numpy())
                labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                predictions.append(model(mfgs, inputs, edge_weights).argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            ben_mask = np.concatenate(ben_mask)
            TP, TN, FP, FN =  evaluation(predictions, labels, ben_mask)
            
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss/len(train_loader):.3f}',
                    '| valid:', f'TP:{TP:>3}, TN:{TN:>6}, FN:{FN:>3}, FP:{FP:>6}',
                  )
            
            if select_model == 'TPFP':
                if TP >= best_tp and FP <= best_fp and epoch>0 and FP<TN:
                    best_tp = TP
                    best_fp = FP
                    torch.save(model.state_dict(), model_pth)
                    print(f'*NOTICE*: best model update! TP:{TP}, FP:{FP}')
                    is_save_model = True
                    
            elif select_model == 'score':
                tp_weight, fp_weight = (TN+FP)/(TP+FN), 1
                score = TP*tp_weight - FP*fp_weight
                if score>best_score:
                    print(f'*NOTICE*: best model update! TP:{TP}, FP:{FP}, score:{score}')
                    best_score = score
                    torch.save(model.state_dict(), model_pth)
                    is_save_model = True
            elif 'score' in select_model:
                args = select_model.split('/')
                div = int(args[1])
                if len(args) == 3: # score/x/y 
                    min_TP = int(args[2])
                elif len(args) == 2: # score/x
                    min_TP = 1
                    
                tp_weight, fp_weight = (TN+FP)/(TP+FN)/float(div), 1
                score = TP*tp_weight - FP*fp_weight
                if score>best_score and TP >= min_TP:
                    print(f'*NOTICE*: best model update! TP:{TP}, FP:{FP}, score:{score}')
                    best_score = score
                    torch.save(model.state_dict(), model_pth)
                    is_save_model = True
                else:
                    print(f'(not update, score is {score}, best score is {best_score}, TP:{TP})')
            else:
                raise NotImplementedError
    
    if not is_save_model:
        print(f'**WARNING**: CANNOT choose best model during training, save the last iteration model')
        torch.save(model.state_dict(), model_pth) 


def exp001():
    save_pth_prefix = 'theia_train'
    threatrace_parse_pth = f'/data1/winsen/threaTrace/graphchi-cpp-master/graph_data/darpatc/{save_pth_prefix}.txt'
    parse_graph_darpa(threatrace_parse_pth, save_pth_prefix)
def exp002():
    save_pth_prefix = 'theia_test'
    threatrace_parse_pth = f'/data1/winsen/threaTrace/graphchi-cpp-master/graph_data/darpatc/{save_pth_prefix}.txt'
    parse_graph_darpa(threatrace_parse_pth, save_pth_prefix)
def exp003(): # try to train threatrace
    save_pth_prefix = 'theia'
    G = pkl.load(open('ATLAS/save/theia_nxG.pkl', 'rb'))
    meta = pkl.load(open('ATLAS/save/theia_meta.pkl', 'rb'))
    groundtruth_pth = '/Volumes/data/数据集/darpa-tc/groundtruth/theia.txt'
    mal_nids, mal_nidnos = get_mal_nid_darpa(G,groundtruth_pth,meta['nid2no'])
    node_data = np.load('ATLAS/save/theia_data.npz')
    dg = nx2dglGraph_darpa(G, meta, node_data, mal_nids)
    train_loader = get_dglLoder_darpa(dg, is_train=True)
    train_loaders = [train_loader]
    test_loader = get_dglLoder_darpa(dg, is_train=False)
    model_pth = 'ATLAS/save/theia_001.pt'
    train_threatrace_darpa(train_loaders, test_loader, model_pth, meta, 
                        select_model='score/3',
                        is_weighted=True, 
                        steps=steps,
                        lr=lr)
def exp004(): # parse trace-test
    save_pth_prefix = 'trace_test'
    threatrace_parse_pth = f'/data1/winsen/threaTrace/graphchi-cpp-master/graph_data/darpatc/{save_pth_prefix}.txt'
    parse_graph_darpa(threatrace_parse_pth, save_pth_prefix)
def exp004(): # parse trace-test
    save_pth_prefix = 'trace_test'
    threatrace_parse_pth = f'/data1/winsen/threaTrace/graphchi-cpp-master/graph_data/darpatc/{save_pth_prefix}.txt'
    parse_graph_darpa(threatrace_parse_pth, save_pth_prefix)
if __name__ == "__main__":
    # exp001()
    # exp002()
    exp003()
    # exp004()