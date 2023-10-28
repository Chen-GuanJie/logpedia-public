from typing import Callable, Dict, List, Optional, Tuple, Union
import random
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl
from dgl.nn import SAGEConv

# from beagle.datasources import SysmonEVTX
# import beagle

# import read_ALTAS_graph as atlas_utils

device = torch.device("cpu")
steps = 1000  # 100
lr = 0.0002  # 0.0001
batch_size = 512
hidden_layer = 32

num_features = None
num_classes = None


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
        h = self.conv2(mfgs[1], (h, h_dst), edge_weight=edge_weights[1])  # <---
        return F.log_softmax(h, dim=1)


## get mal node id from keyword of ALTAS
def get_mal_node_id(G: nx.MultiDiGraph, mal_keyword: List[str]) -> List[str]:
    mal_node_cnt = 0
    mal_nid = []

    for i, label in enumerate(G.nodes()):
        for mal in mal_keyword:
            if mal in label:
                mal_nid.append(label)
                mal_node_cnt += 1
                break
    print("mal_node_num:", mal_node_cnt)
    return mal_nid


no2nid_map = {}
node_map = {}


# Get train-/test-able DGLGraph from RAW nxGraph
def nxGraph2DGLGraph(
    G: nx.MultiDiGraph, mal_keyword: List[str], graph_id=1
) -> dgl.DGLGraph:
    global num_features
    global num_classes
    global no2nid_map
    elabel2no, elabel_cnt = {}, 0  # edge_type to no. map
    for h, t, r in G.edges(data=True):
        if r["relationship"] not in elabel2no:
            elabel2no[r["relationship"]] = elabel_cnt
            elabel_cnt += 1

    nid2no, no2nid = {}, {}  # node_id to no. map
    nlabel2no, nlabel_cnt = {}, 0  # node_type to no. map

    n_cnt = len(G.nodes())
    X, y = np.zeros((n_cnt, elabel_cnt * 2)), []  # GNN feat. and label
    for i, (nid, nattr) in enumerate(G.nodes(data=True)):
        nid2no[nid] = i
        no2nid[i] = nid
        if nattr["type"] not in nlabel2no:
            nlabel2no[nattr["type"]] = nlabel_cnt
            nlabel_cnt += 1
        y.append(nlabel2no[nattr["type"]])

        for _, _, r in G.in_edges(nid, data=True):
            X[i][elabel2no[r["relationship"]]] += 1
        for _, _, r in G.out_edges(nid, data=True):
            X[i][elabel2no[r["relationship"]] + elabel_cnt] += 1
    no2nid_map[graph_id] = no2nid
    print("elabel_cnt", elabel_cnt, "nlabel_cnt", nlabel_cnt)
    if num_features is None and num_features is None:
        num_classes, num_features = nlabel_cnt, elabel_cnt * 2
        print(f"** GLOBAL: num_classes={num_classes}, num_features={num_features} **")
    else:
        if nlabel_cnt != num_classes or elabel_cnt * 2 != num_features:
            print("FATAL ERROR! INCONSIST num_classes/num_features!")
            exit(-1)

    s_edges, e_edges = [], []
    edge_weights = []
    # G = nx.relabel_nodes(G,nid2no)
    edge_type_no = []
    H = nx.DiGraph(G)
    for u, v, attr in H.edges(data=True):
        s_edges.append(nid2no[u])
        e_edges.append(nid2no[v])
        edge_count = G.number_of_edges(u, v)
        edge_weights.append(edge_count)
        edge_type_no.append(elabel2no[attr["relationship"]])

    dg = dgl.graph((s_edges, e_edges))
    dg.ndata["feat"] = torch.tensor(X, dtype=torch.float)
    dg.ndata["label"] = torch.tensor(y, dtype=torch.long)
    dg.edata["weight"] = torch.tensor(edge_weights, dtype=torch.float)
    dg.edata["etype"] = torch.tensor(edge_type_no, dtype=torch.long)

    benign_mask = np.asarray([True] * X.shape[0])
    mal_nid = get_mal_node_id(G, mal_keyword)
    for id in mal_nid:
        benign_mask[nid2no[id]] = False
    print("Benign node num:", benign_mask.sum())
    node_map[graph_id] = (len(mal_nid), benign_mask.sum())

    dg.ndata["ben_mask"] = torch.tensor(benign_mask, dtype=torch.bool)
    return dg


def nxGraph2Loder(
    G: nx.MultiDiGraph, mal_keyword: List[str], is_train=True, graph_id=1
) -> dgl.dataloading.DataLoader:
    print("\nProcess Graph:", G.name)

    dg = nxGraph2DGLGraph(G, mal_keyword, graph_id)

    sampler = dgl.dataloading.NeighborSampler([-1, -1])
    if is_train:
        print("generate TRAINING loader")
        train_dataloader = dgl.dataloading.DataLoader(
            # The following arguments are specific to DGL's DataLoader.
            dg,  # The graph
            dg.nodes()[
                dg.ndata["ben_mask"]
            ],  # The node IDs to iterate over in minibatches
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
        print("generate TESTING loader")
        valid_dataloader = dgl.dataloading.DataLoader(
            dg,
            dg.nodes(),
            sampler,
            batch_size=100000,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            device=device,
        )
        return valid_dataloader


def evaluation(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    benign_mask: np.ndarray,
):
    TN = np.sum((y_pred == y_true) * benign_mask)
    TP = np.sum((y_pred != y_true) * (1 - benign_mask))
    FP = np.sum(benign_mask) - TN
    FN = np.sum((1 - benign_mask)) - TP

    # total = TP + TN + FP + FN

    # ben_class_dist = []
    # pred = y_pred[benign_mask]
    # for i in range(7):
    #     ben_class_dist.append(len(pred[pred==i]))

    # mal_class_dist = []
    # pred = y_pred[np.asarray(1-benign_mask).astype(np.bool)]
    # for i in range(7):
    #    mal_class_dist.append(len(pred[pred==i]))

    # print(f'(ben_class_dist: {ben_class_dist}, mal_class_dist: {mal_class_dist}')

    # return f'TP:{TP:>3}, TN:{TN:>6}, FN:{FN:>3}, FP:{FP:>6} (Total:{total:>6})'

    return TP, TN, FP, FN


# get class weight for imbalance dataset.
def get_loss_weight(
    y_train: torch.Tensor,
) -> torch.Tensor:
    num = []
    weight = []
    y_train = y_train.numpy()
    for i in range(num_classes):
        num.append(len(y_train[y_train == i]))
        if num[i] == 0:
            weight.append(0.0)
        else:
            weight.append(len(y_train) / num[i])
    print("Train Each Class Num:", num)
    print("Weight Each Class Num:", weight)
    return torch.tensor(weight, dtype=torch.float).to(device)


def train_single_model(
    train_loaders: List[dgl.dataloading.DataLoader],
    valid_dataloader: dgl.dataloading.DataLoader,
    model_pth: str,
    select_model="TPFP",  # 'TPFP' save model when TP and FP both become better;
    # 'score': weight heuristic score TP/TN weight is Neg/Pos,
    # 'score/x': heuristic score /x common is 2 or 3 (lower the weight on TP, its okay if TP reduced a little but FP reduce more)
    is_weighted=False,
    steps=steps,
    lr=lr,
    check_itv=10,  # interval epoches to check whether best model
    dont_train=False,
):
    # test_dg:dgl.DGLGraph,

    model = Model(num_features, hidden_layer, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    if dont_train:
        return model
    if select_model == "TPFP":
        best_tp, best_fp = 0, np.inf  # use for save best model
    elif "score" in select_model:
        best_score = -np.inf

    if is_weighted:
        labels = []
        for _, _, mfgs in valid_dataloader:
            labels = mfgs[-1].dstdata["label"]
        class_weight = get_loss_weight(labels)

    is_save_model = False
    for epoch in range(steps):
        model.train()
        total_loss = 0

        for train_loader in train_loaders:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(train_loader):
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]
                edge_weights = [mfgs[0].edata["weight"], mfgs[1].edata["weight"]]

                out = model(mfgs, inputs, edge_weights)

                if is_weighted:
                    loss = F.nll_loss(out, labels, weight=class_weight)
                else:
                    loss = F.nll_loss(out, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

        if (epoch == steps - 1) or (epoch % check_itv == 0 and epoch > 0):
            model.eval()
            predictions = []
            labels = []
            ben_mask = []
            for input_nodes, output_nodes, mfgs in valid_dataloader:
                inputs = mfgs[0].srcdata["feat"]
                edge_weights = [mfgs[0].edata["weight"], mfgs[1].edata["weight"]]
                # ben_mask.append(mfgs[0].srcdata["ben_mask"].cpu().numpy())
                ben_mask.append(mfgs[-1].dstdata["ben_mask"].cpu().numpy())
                labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                predictions.append(
                    model(mfgs, inputs, edge_weights).argmax(1).cpu().numpy()
                )
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            ben_mask = np.concatenate(ben_mask)
            TP, TN, FP, FN = evaluation(predictions, labels, ben_mask)

            print(
                f"Epoch {epoch:>3} | Train Loss: {total_loss/len(train_loader):.3f}",
                "| valid:",
                f"TP:{TP:>3}, TN:{TN:>6}, FN:{FN:>3}, FP:{FP:>6}",
            )

            if select_model == "TPFP":
                if TP >= best_tp and FP <= best_fp and epoch > 0 and FP < TN:
                    best_tp = TP
                    best_fp = FP
                    torch.save(model.state_dict(), model_pth)
                    print(f"*NOTICE*: best model update! TP:{TP}, FP:{FP}")
                    is_save_model = True

            elif select_model == "score":
                tp_weight, fp_weight = (TN + FP) / (TP + FN), 1
                score = TP * tp_weight - FP * fp_weight
                if score > best_score:
                    print(
                        f"*NOTICE*: best model update! TP:{TP}, FP:{FP}, score:{score}"
                    )
                    best_score = score
                    torch.save(model.state_dict(), model_pth)
                    is_save_model = True
            elif "score" in select_model:
                div = int(select_model[-1])
                tp_weight, fp_weight = (TN + FP) / (TP + FN) / float(div), 1
                score = TP * tp_weight - FP * fp_weight
                if score > best_score:
                    print(
                        f"*NOTICE*: best model update! TP:{TP}, FP:{FP}, score:{score}"
                    )
                    best_score = score
                    torch.save(model.state_dict(), model_pth)
                    is_save_model = True
            else:
                raise NotImplementedError

    if not is_save_model:
        print(
            f"**WARNING**: CANNOT choose best model during training, save the last iteration model"
        )
        torch.save(model.state_dict(), model_pth)
    return model
