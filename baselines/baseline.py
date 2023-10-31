import os
import networkx as nx
import pandas as pd
from datetime import datetime

ROOT_PATH = "" #set it as the root path of this repository
LOG_DIR = f"{ROOT_PATH}/data/attack-scenario"
SCENARIO_NUM = 10


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_all_graph(file_list, consolidate=True):
    seen_entity = []
    g = nx.MultiDiGraph()

    def get_graph(df: pd.DataFrame, consolidate=True):
        for _, row in df.iterrows():
            source_pid = row["source_pid"]
            des_pid = row["des_pid"]
            timestamp = datetime.timestamp(
                datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S.%f")
            )
            relationship = row["relationship"]
            source = row["source"].lower().replace("\\", "/")
            destination = row["destination"].lower().replace("\\", "/")

            s_img = f"{source}_{source_pid}"
            s_id = s_img
            if source_pid == des_pid:
                d_img = destination
            else:
                d_img = f"{destination}_{des_pid}"
            d_id = d_img
            g.add_edge(
                s_id,
                d_id,
                key=timestamp if consolidate else None,
                timestamp=timestamp,
                relationship=relationship,
                source=s_img,
                destination=d_img,
            )

            if d_id not in seen_entity:
                seen_entity.append(d_id)
                g.add_node(d_id, img=d_img, id=len(seen_entity) - 1)

            if s_id not in seen_entity:
                seen_entity.append(s_id)
                g.add_node(
                    s_id,
                    img=s_img,
                    id=len(seen_entity) - 1,
                    type="process",
                    pid=source_pid,
                )

            if "reg" in relationship:
                g.nodes[d_id]["type"] = "registry_key"
            elif relationship == "load" or "file" in relationship:
                g.nodes[d_id]["type"] = "file"
            elif relationship == "access" or relationship == "process_create":
                g.nodes[d_id]["type"] = "process"
                g.nodes[d_id]["pid"] = des_pid
            else:
                ...
        pass

    for f in file_list:
        get_graph(pd.read_csv(f), consolidate)
    return g


def count_entites(g: nx.MultiDiGraph, mal_label=[]):
    mal_node = []
    mal_edge = []
    for n, d in g.nodes(data=True):
        for ml in mal_label:
            if ml in d["img"]:
                mal_node.append(n)
                break
    for u, v in g.edges():
        if u in mal_node or v in mal_label:
            mal_edge.append((u, v))
    return mal_node, mal_edge


def load_all_file(scenario, consolidate=True):
    # scenario: scenario id
    def split_file_name(log_file_name: str):
        mal_dll = None
        ls = log_file_name.split("_")
        if ls[0] == "ByPassUAC":
            label_file = f"ByPassUAC_{ls[1]}"
        elif "(sysmon)" in ls[0]:
            label_file = ls[0].split("(sysmon)")[0]
        else:
            label_file = ls[0]
        for i in range(1, len(ls)):
            if ".exe" in ls[i]:
                mal_process = ls[i].split("(")[0]
            elif ".dll" in ls[i]:
                mal_dll = ls[i]
        return label_file, mal_process, mal_dll

    files = []
    mal_label = []
    for log_file in os.scandir(f"{LOG_DIR}/scenario{scenario}"):
        log_file_name, _ = os.path.splitext(log_file.name)
        label_file, mal_process, mal_dll = split_file_name(log_file_name)
        if mal_dll is not None:
            mal_label.append(mal_dll)
        files.append(log_file.path)
        with open(
            f"{LOG_DIR}/label/{label_file}(sysmon).txt", "r", encoding="utf-8"
        ) as f:
            for line in f.readlines():
                if "pid:" in line:
                    mal_label.append(f"{mal_process}_{line.strip().split(':')[-1]}")
                else:
                    mal_label.append(line.strip())
        mal_label = list(set([m.lower().replace("\\", "/") for m in mal_label]))
    return get_all_graph(files, consolidate), mal_label


def fix_log_timeline(i):
    files = []

    for log_file in os.scandir(f"{LOG_DIR}/scenario{i}"):
        files.append(log_file.path)
    for f in files:
        df = pd.read_csv(f)
        timeline = df["time"]
        d = timeline.apply(
            lambda t: datetime.timestamp(datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")),
            0,
        )
        a = d[d.diff() < 0].index - 1
        mi = a - 1
        ma = a + 1
        d[a] = (d[mi].values + d[ma].values) / 2
        d.describe().round(2)
        b = d[d.diff() < 0].index
        pass


def keep_process(mal_node_ids, process_only):
    if not process_only:
        return mal_node_ids
    mal_process = []
    for m in mal_node_ids:
        if ".exe_" in m:
            mal_process.append(m)
    return mal_process


def count_entities(process_only=True):
    for i in range(1, SCENARIO_NUM + 1):
        g, l = load_all_file(i)
        mal_node_ids = keep_process(l, process_only)
        mal_node, mal_edge = count_entites(g, mal_node_ids)
        print(
            f"{i} == good node:{g.number_of_nodes()-len(mal_node)} good edge:{g.number_of_edges()-len(mal_edge)}"
        )
        print(f"mal node:{len(mal_node)}  mal edge:{len(mal_edge)}")
        return {
            "node": g.number_of_nodes() - len(mal_node),
            "good_edge": g.number_of_edges() - len(mal_edge),
        }
        # with open(
        #     f"{ROOT_PATH}/data/attack-scenario/mal_node/{i}.txt",
        #     "w",
        #     encoding="utf-8",
        # ) as f:
        #     f.writelines("\n".join(mal_node))


def calc(df: pd.DataFrame):
    df["accuracy"] = (df["TP"] + df["TN"]) / (df["TP"] + df["TN"] + df["FP"] + df["FN"])
    df["precision"] = df["TP"] / (df["TP"] + df["FP"])
    df["recall"] = df["TP"] / (df["TP"] + df["FN"])
    return df
