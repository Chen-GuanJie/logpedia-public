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

    pkl.dump(G, open(f'ATLAS/save/{save_pth_prefix}_nxG.pkl', 'wb'))
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
    np.savez_compressed(f'ATLAS/save/{save_pth_prefix}_data.npz', X=X, y=y)
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
    pkl.dump(meta, open(f'ATLAS/save/{save_pth_prefix}_meta.pkl', 'wb'))
    if verbose: print(f'(Info: successfully save meta infos at {save_pth_prefix}_meta.pkl')
        
        
def exp001():
    save_pth_prefix = 'theia'
    parse_pth = '/Volumes/data/数据集/darpa-tc/ta1-theia-e3-official-6r.json/ta1-theia-e3-official-6r.json.txt'
    parse_graph_darpa(parse_pth, save_pth_prefix)

if __name__ == "__main__":
    exp001()
