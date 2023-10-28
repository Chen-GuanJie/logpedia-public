from baseline import *


def prov(num=2000, window=1000):
    relation = {}

    def convert(g: nx.MultiDiGraph, path):
        s = ""
        for i in range(len(path) - 1):
            s_type = g.nodes[path[i]]["type"]
            name = path[i].split("/")[-1].split("_")[0]
            rel = relation[(path[i], path[i + 1])]
            s = f"{s} {s_type}:{name} {rel}"
        s = s[1:]
        end_type = g.nodes[path[-1]]["type"]
        end_name = path[-1].split("/")[-1]
        return f"{s} {end_type}:{end_name}"

    from get_k_path import get_k_path
    import matplotlib.pyplot as plt

    for i in range(1, SCENARIO_NUM + 1):
        g, l = load_all_file(i, True)
        for u, v, d in g.edges(data=True, keys=False):
            relation[(u, v)] = d["relationship"]

        nodes = [n for n, i in list(g.in_degree()) if i > 0]
        for node in nodes:
            in_edges = list(g.in_edges(nbunch=node, data=True, keys=True))
            min_in_time = 999999999999999
            pce = None
            for u, v, k, d in in_edges:
                if min_in_time > d["timestamp"]:
                    min_in_time = d["timestamp"]
                if d["relationship"] == "process_create":
                    pce = (u, v, k, d["timestamp"])
            if pce is not None and pce[3] >= min_in_time:
                g.edges[(pce[0], pce[1], pce[2])]["timestamp"] = round(
                    min_in_time - 0.001, 3
                )

        kp, allp = get_k_path(g, num, window)
        kps = [convert(g, p) for p in kp]
        kps = list(set(kps))

        bad = []
        ap = []
        for p in allp:
            t = "".join(p)
            good = True
            for label in l:
                if label in t:
                    bad.append(p)
                    good = False
            if good:
                ap.append(p)
        ap = [convert(g, p) for p in ap]
        ap = list(set(ap))
        bad = [convert(g, p) for p in bad]
        bad = list(set(bad))

        with open(
            f"{ROOT_PATH}/sample-enterprise-data/benign-fv.csv",
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines("\n".join(ap))
        with open(
            f"{ROOT_PATH}/sample-enterprise-data/anomaly-fv.csv",
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines("\n".join(bad))
