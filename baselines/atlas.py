from baseline import *


atlas_dir = f"{ROOT_PATH}/baselines/ATLAS"


def get_atlas_data(
    g: nx.MultiDiGraph, lables=[], file="", lable_file="malicious_labels.txt"
):
    # eg:28997136,,,1736,1716,c:/windows/explorer.exe,,,,,,,,,,,,,,-LA-
    preprocessing_lines = []
    for u, v, d in sorted(g.edges(data=True), key=lambda x: x[2]["timestamp"]):
        timestamp = d["timestamp"]
        in_e = list(g.in_edges(nbunch=u))
        if in_e:
            parent_process_id = g.nodes[in_e[0][0]]["pid"]
        else:
            parent_process_id = ""
        process_name = g.nodes[u]["img"].split(".exe_")[0] + ".exe"
        accesses = d["relationship"]
        if g.nodes[v]["type"] == "process":
            obj_name = g.nodes[v]["img"].split(".exe_")[0] + ".exe"
        else:
            obj_name = g.nodes[v]["img"]
        process_id = g.nodes[u]["img"].split(".exe_")[-1]
        line = f"{timestamp},,,{process_id},{parent_process_id},{process_name},,,,,,,,,,,,{accesses},{obj_name}"
        preprocessing_lines.append("\n" + line.lower().replace("\\", "/") + ",-LA-")
    lable_lines = []
    for u, d in g.nodes(data=True):
        for lable in lables:
            if lable in d["img"]:
                lable_lines.append(d["img"] + "\n")
    with open(file, "w", encoding="utf-8") as f:
        f.writelines(preprocessing_lines)
    with open(lable_file, "w", encoding="utf-8") as f:
        f.writelines(lable_lines)


def optimize_graph(g: nx.MultiDiGraph):
    new_g = nx.DiGraph()
    for u, v, k, d in g.edges(data=True, keys=True):
        if new_g.has_edge(u, v) and new_g.edges[u, v]["timestamp"] < k:
            continue
        new_g.add_edge(u, v, **d)
        new_g.add_nodes_from([(u, g.nodes[u]), (v, g.nodes[v])])
    return new_g


def convert_to_atlas_graph(g: nx.MultiDiGraph):
    new_g = nx.MultiDiGraph()
    for n, d in g.nodes(data=True):
        new_g.add_node(n, type=d["type"])
    for u, v, d in g.edges(data=True):
        new_g.add_edge(
            u,
            v,
            capacity=1.0,
            label=d["relationship"] + "_" + str(d["timestamp"]),
            type=d["relationship"],
            timestamp=d["timestamp"],
        )
        new_g.nodes[u]["timestamp"] = d["timestamp"]
        new_g.nodes[v]["timestamp"] = d["timestamp"]
    return new_g


def graph_read(g, output_file_path):
    written_lines = []
    G = convert_to_atlas_graph(optimize_graph(g))
    with open(output_file_path, "a", encoding="utf-8") as output_file:
        for a, b, data in sorted(G.edges(data=True), key=lambda x: x[2]["timestamp"]):
            op_type = data["type"]
            formatted_str = "{a} {w} {b}\n".format(
                a=a.lstrip().rstrip().replace(" ", ""),
                w=op_type,
                b=b.lstrip().rstrip().replace(" ", ""),
            )
            if not formatted_str in written_lines:
                output_file.write(formatted_str)
                written_lines.append(formatted_str)


def prepare_atlas(process_only=True):
    # atlas_dir = f"{ROOT_PATH}/baselines/ATLAS"
    check_dir(f"{atlas_dir}/output")
    check_dir(f"{atlas_dir}/training_logs")
    for i in range(1, SCENARIO_NUM + 1):
        g, l = load_all_file(i)
        l = keep_process(l, process_only)
        graph_read(
            g,
            output_file_path=f"{atlas_dir}/output_backup/seq_graph_testing_preprocessed_logs_{i}.dot.txt",
        )
        get_atlas_data(g, file=f"{atlas_dir}/output/testing_preprocessed_logs_{i}")
        check_dir(f"{atlas_dir}/training_logs/{i}")
        check_dir(f"{atlas_dir}/testing_logs/{i}")

        with open(
            f"{atlas_dir}/training_logs/{i}/malicious_labels.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines("\n".join(l))
        with open(
            f"{atlas_dir}/testing_logs/{i}/malicious_labels.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines("\n".join(l))


def train_atlas(train_id=1):
    atlas_Dir = atlas_dir.replace("/", "\\")
    source_dir = f"{atlas_Dir}\output_backup"
    des_dir = f"{atlas_Dir}\output"
    train_seq_file = f"seq_graph_training_preprocessed_logs_{train_id}.dot.txt"
    os.chdir(atlas_Dir)
    os.system(
        f"copy  {source_dir}\{train_seq_file.replace('train','test')} {des_dir}\{train_seq_file}"
    )
    os.system(f"python {atlas_Dir}/atlas.py --do_train 1")
    os.system(f"python {atlas_Dir}/atlas.py --do_train 1 --load_resampling 1")
    os.remove(f"{des_dir}/{train_seq_file}")


def test_atlas():
    atlas_Dir = atlas_dir.replace("/", "\\")
    source_dir = f"{atlas_Dir}\output_backup"
    des_dir = f"{atlas_Dir}\output"
    os.chdir(atlas_Dir)
    for test in range(1, SCENARIO_NUM + 1):
        test_seq_file = f"seq_graph_testing_preprocessed_logs_{test}.dot.txt"
        os.system(f"copy  {source_dir}\{test_seq_file} {des_dir}\{test_seq_file}")
        with open(
            f"{atlas_Dir}\\testing_logs\{test}\malicious_labels.txt",
            encoding="utf-8",
        ) as f:
            mal_num = len(f.readlines())
        for mal in range(mal_num):
            os.system(f"python {atlas_Dir}/atlas.py --test {test} --mal {mal}")
        os.remove(f"{des_dir}/{test_seq_file}")


def analsys_atlas(index=None, process_only=True):
    def calc_fnfp(g: nx.MultiDiGraph, mal_node_ids, result_nodes):
        malicious_nodes = " ".join(result_nodes)
        tp = 0
        fn_nodes = []
        for mn in mal_node_ids:
            if mn in malicious_nodes:
                tp = tp + 1
            else:
                fn_nodes.append(mn)
        fp = len(result_nodes) - tp
        mal_process_num = len([n for n in g.nodes if ".exe_" in n])
        tn = mal_process_num - len(mal_node_ids) - fp
        fn = len(mal_node_ids) - tp
        fp_nodes = []
        for n in result_nodes:
            is_fp = True
            for m in mal_node_ids:
                if m in n:
                    is_fp = False
            if is_fp:
                fp_nodes.append(n)
        return tp, tn, fp, fn, fp_nodes, fn_nodes

    df = pd.DataFrame(columns=["scenario", "user_artifact", "TP", "TN", "FP", "FN"])

    result = {}
    for i in range(1, SCENARIO_NUM + 1):
        result[i] = {}
        if index is not None and i != index:
            continue
        g, l = load_all_file(i)

        mal_node_ids = keep_process(l, process_only)
        mal_node, _ = count_entites(g, mal_node_ids)

        for file in os.scandir(f"{atlas_dir}/result"):
            fn_split = file.name.split("_")
            if i == int(fn_split[1]):
                j = int(fn_split[-1].split(".")[0])
                with open(file, encoding="utf-8") as f:
                    r = f.readlines()
                    result_nodes = [x.strip() for x in r if ".exe_" in x]

                tp, tn, fp, fn, fp_nodes, fn_nodes = calc_fnfp(
                    g, mal_node, result_nodes
                )
                result[i][result_nodes[j]] = {}
                result[i][result_nodes[j]]["result"] = (tp, tn, fp, fn)
                result[i][result_nodes[j]]["False Positives"] = fp_nodes
                result[i][result_nodes[j]]["False Negatives"] = fn_nodes
                df.loc[len(df.index)] = [i, result_nodes[j], tp, tn, fp, fn]
    df = calc(df)
    save_path = f"{ROOT_PATH}/baselines/result/atlas"
    check_dir(f"{ROOT_PATH}/baselines/result")
    check_dir(save_path)
    df.to_csv(f"{save_path}/result.csv")

    import json

    with open(f"{save_path}/result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


dirs = [
    "output",
    "output_backup",
    "training_logs",
    "testing_logs",
    "resampling",
    "result",
]
for dir in dirs:
    check_dir(f"{atlas_dir}/{dir}")
if __name__ == "__main__":
    prepare_atlas()
    train_atlas()
    test_atlas()
    analsys_atlas()
