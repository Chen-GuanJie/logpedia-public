# feature + edge attack find real nodes to link
from typing import Callable, Dict, List, Optional, Tuple, Union
import random
import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from pyvis.network import Network

# get malicious node id from G and keyword
def get_mal_node_id(G:nx.MultiDiGraph,
                    mal_keyword:List[str]
                    ) -> (List[str]):
    
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


# parse G to several format feature and labels
def parse_graph(G:nx.MultiDiGraph, mal_keyword:List[str]):
    elabel2no, no2elabel, elabel_cnt = {}, {}, 0 # edge_type to no. map
    for h, t, r in G.edges(data=True):
        if r['type'] not in elabel2no:
            elabel2no[r['type']] = elabel_cnt
            no2elabel[elabel_cnt] = r['type']
            elabel_cnt += 1
            

    nid2no, no2nid = {},{} # node_id to no. map
    nlabel2no, no2nlabel, nlabel_cnt = {}, {}, 0 # node_type to no. map

    n_cnt = len(G.nodes())
    X, y = np.zeros((n_cnt,elabel_cnt*2)), [] # GNN feat. and label
    for i, (nid, nattr) in enumerate(G.nodes(data=True)):
        nid2no[nid] = i
        no2nid[i] = nid
        if nattr['type'] not in nlabel2no:
            nlabel2no[nattr['type']] = nlabel_cnt
            no2nlabel[nlabel_cnt] = nattr['type']
            nlabel_cnt += 1
        y.append(nlabel2no[nattr['type']])
        
        for _,_,r in G.in_edges(nid, data=True):
            X[i][elabel2no[r['type']]] += 1
        for _,_,r in G.out_edges(nid, data=True):
            X[i][elabel2no[r['type']]+elabel_cnt] += 1
            
    print('elabel_cnt', elabel_cnt, 'nlabel_cnt', nlabel_cnt)
    
    mal_nids = get_mal_node_id(G, mal_keyword)
    mal_nidnos = list(map(nid2no.get, mal_nids))
    
    return X, y, elabel2no, nlabel2no, nid2no, no2nid, mal_nids, mal_nidnos, no2nlabel, no2elabel
