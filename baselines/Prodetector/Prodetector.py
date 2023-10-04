import networkx as nx
import pandas as pd
from math import log2
from typing import List, Tuple

import Prodetector.eppstein as ep

__all__ = ["get_k_path"]
_PSEUDO_SOURCE = -1
_PSEUDO_SINK = -2


def __get_new_node(g: nx.MultiDiGraph):
    i = max(list(g.nodes(data=False)))
    if i > 999999:
        i = 1
    while True:
        i = i + 1
        yield i


def _add_new_node(g: nx.MultiDiGraph, new_node, father, new_edges) -> None:
    g.add_node(new_node, **g.nodes[father], father=father)
    g.add_edges_from(new_edges)


def _calc_regularity_score(g: nx.MultiDiGraph, window) -> None:
    if not nx.get_edge_attributes(g, "time"):
        edge_attrs = {}
        for u, v, k, d in list(g.edges(data=True, keys=True)):
            edge_attrs[(u, v, k)] = {"time": d["data"]["timestamp"]}
        nx.set_edge_attributes(g, edge_attrs)
    attrs = {}
    for node in g.nodes(data=False):
        # calculate IN and OUT for a node
        in_times = [d["time"] for _, _, d in list(g.in_edges(nbunch=node, data=True))]
        out_times = [d["time"] for _, _, d in list(g.out_edges(nbunch=node, data=True))]
        if in_times:
            i, _ = pd.cut(x=in_times, bins=window, retbins=True, labels=False)
            in_score = window - len(set(i)) + 1  # 1 - len(set(i)) / window
        else:
            in_score = 1 + window
        if out_times:
            o, _ = pd.cut(x=out_times, bins=window, retbins=True, labels=False)
            out_score = window - len(set(o)) + 1  # 1 - len(set(o)) / window
        else:
            out_score = 1 + window
        attrs[node] = {"in": in_score, "out": out_score}
    nx.set_node_attributes(g, attrs)
    attrs = {}
    for u, v, k in g.edges(keys=True):
        # calculate regularity score of event
        attrs[(u, v, k)] = {"weight": log2(g.nodes[u]["out"]) + log2(g.nodes[v]["in"])}
    nx.set_edge_attributes(g, attrs)


def get_k_path(g: nx.MultiDiGraph, num=1, window=0) -> List[Tuple[List[int], float]]:
    # g = nx.MultiDiGraph(dg)
    next_node = __get_new_node(g)
    _calc_regularity_score(g, window)
    nodes = [n for n, i in list(g.in_degree()) if i > 0]
    for node in nodes:
        in_edges = list(g.in_edges(nbunch=node, data=True, keys=False))
        in_times = [d["time"] for _, _, d in in_edges]
        min_in_time = min(in_times)
        out_edges = list(g.out_edges(nbunch=node, data=True, keys=False))
        if g.out_degree(node) > 0:
            # some out_edges are earlier than all in_edges, make a new node with only out_edges
            min_out_time = min([d["time"] for _, _, d in out_edges])
            if min_in_time > min_out_time:
                new_node = next(next_node)
                out_before = [
                    (new_node, v, d) for _, v, d in out_edges if d["time"] < min_in_time
                ]
                _add_new_node(g, new_node, node, out_before)
        if g.in_degree(node) == 1 or len(set(in_times)) <= 1:
            continue
        seen_time = {}
        for f, _, d in in_edges:  # split nodes
            if (f, d["time"]) in seen_time.keys():
                g.add_edge(f, seen_time[(f, d["time"])], **d)
                continue
            new_node = next(next_node)
            seen_time[(f, d["time"])] = new_node
            out_later = [
                (new_node, v, od) for _, v, od in out_edges if od["time"] > d["time"]
            ]
            _add_new_node(g, new_node, node, [(f, new_node, d)] + out_later)
        g.remove_node(node)
    # converts g to a single source and single sink flow graph
    in_degree_0 = [(_PSEUDO_SOURCE, n) for n, i in list(g.in_degree()) if i == 0]
    out_degree_0 = [(n, _PSEUDO_SINK) for n, i in list(g.out_degree()) if i == 0]
    g.add_edges_from(in_degree_0 + out_degree_0, weight=0)
    kp = ep.k_shortest_paths(nx.DiGraph(g), _PSEUDO_SOURCE, _PSEUDO_SINK, num)
    for j, path in enumerate(kp):
        for i, node in enumerate(path[0]):
            if "father" in g.nodes[node]:
                kp[j][0][i] = g.nodes[node]["father"]
    return kp

