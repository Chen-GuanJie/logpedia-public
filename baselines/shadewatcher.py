from baseline import *


def get_shadewatcher_data(
    g: nx.MultiDiGraph, entities: list = [], output_folder="output"
):
    """inter_ids : process id to process"""
    inter_ids = []
    check_dir(output_folder)
    for n, d in g.nodes(data=True):
        for p in entities:
            if p in d["img"]:
                inter_ids.append(n)

    entity = {}  # hash_id to id
    with open(f"{output_folder}/entity2id.txt", "w", encoding="utf-8") as f:
        f.write("{a}\n".format(a=g.number_of_nodes()))
        id2hash = [0] * g.number_of_nodes()
        for a, data in g.nodes(data=True):
            entity[a] = data["id"]
            id2hash[data["id"]] = a
        for id in range(len(id2hash)):
            formatted_str = f"{id2hash[id]} {id}\n"
            f.write(formatted_str)

    r = {}
    with open(f"{output_folder}/relation2id.txt", "w", encoding="utf-8") as f:
        relation = []
        for a, b, data in g.edges(data=True):
            c = data["relationship"]
            if c not in relation:
                relation.append(c)
        relation.extend(
            [
                "create",
                "recv",
                "send",
                "mkdir",
                "rmdir",
                "open",
                "load",
                "read",
                "write",
                "connect",
                "getpeername",
                "filepath",
                "mode",
                "mtime",
                "linknum",
                "uid",
                "count",
                "nametype",
                "version",
                "dev",
                "sizebyte",
            ]
        )
        f.write("{a}\n".format(a=len(relation)))
        for i in range(len(relation)):
            r[relation[i]] = i
            formatted_str = f"{relation[i]} {i}\n"
            f.write(formatted_str)

    with open(f"{output_folder}/train2id.txt", "w", encoding="utf-8") as f:
        f.write("{a}\n".format(a=g.number_of_edges()))
        for a, b, data in g.edges(data=True):
            c = data["relationship"]
            formatted_str = f"{entity[a]} {entity[b]} {r[c]}\n"
            f.write(formatted_str)

    with open(f"{output_folder}/inter2id_0.txt", "w", encoding="utf-8") as f:
        for node_id in inter_ids:
            l = list(g.pred[node_id])
            l.extend(list(g.succ[node_id]))
            formatted_str = str(entity[node_id])
            for id in l:
                formatted_str = formatted_str + " " + str(entity[id])
            formatted_str = formatted_str + "\n"
            f.write(formatted_str)


def prepare_shadewatcher(
    process_only=True,
    output_path=f"{ROOT_PATH}/baselines/ShadeWatcher/data/encoding",
):
    for i in range(1, SCENARIO_NUM + 1):
        g, l = load_all_file(i)
        l = keep_process(l, process_only)
        get_shadewatcher_data(
            g,
            entities=l,
            output_folder=f"{output_path}/{i}",
        )


def test_shadewatcher():
    EBD_PATH = f"{ROOT_PATH}/baselines/shadewatcher/embedding"
    ENCODE_PATH = f"{ROOT_PATH}/baselines/ShadeWatcher/data/encoding"
    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import classification_report, confusion_matrix

    np.random.seed(42)
    df = pd.DataFrame(columns=["scenario", "TP", "TN", "FP", "FN"])
    result = {}
    for file in os.scandir(EBD_PATH):
        with open(f"{ENCODE_PATH}/{file.name}/inter2id_0.txt", encoding="utf-8") as f:
            lines = f.readlines()
            mal_id = [int(l.split(" ")[0]) for l in lines]
        with open(f"{ENCODE_PATH}/{file.name}/entity2id.txt", encoding="utf-8") as f:
            lines = f.readlines()
            entities = [l.split(" ")[0] for l in lines]
            entities.pop(0)
        mal_entities = [entities[i] for i in mal_id]
        good_entities = [entities[i] for i in range(len(entities)) if i not in mal_id]
        embedding = np.load(f"{file.path}/transr/embedding/0.001/gnn.npz")
        anomalous_nodes = embedding["entity_attr_embed"][mal_id]
        normal_nodes = np.delete(embedding["entity_attr_embed"], mal_id, 0)
        all_nodes = np.vstack((normal_nodes, anomalous_nodes))
        labels = np.ones(len(all_nodes))
        labels[len(normal_nodes) :] = -1
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(normal_nodes)
        predictions = model.predict(all_nodes)
        # print(file.name)
        # print(classification_report(labels, predictions))
        # print(confusion_matrix(labels, predictions))
        mal_as_mal = predictions[len(normal_nodes) :] == -1
        # mal_as_good = predictions[len(normal_nodes):] == 1
        good_as_mal = predictions[: len(normal_nodes)] == -1
        # good_as_good =predictions[:len(normal_nodes)] == 1

        tp = [
            i
            for i, mal in enumerate(mal_as_mal)
            if mal == True and ".exe_" in mal_entities[i]
        ]  # np.where(mal_as_mal == True)
        fn = [
            i
            for i, mal in enumerate(mal_as_mal)
            if mal == False and ".exe_" in mal_entities[i]
        ]  # np.where(mal_as_mal == False)
        fp = [
            i
            for i, mal in enumerate(good_as_mal)
            if mal == True and ".exe_" in good_entities[i]
        ]  # np.where(good_as_mal == True)
        tn = [
            i
            for i, mal in enumerate(good_as_mal)
            if mal == False and ".exe_" in good_entities[i]
        ] 
        
        print(f"{len(tp)}, {len(tn)}, {len(fp)}, {len(fn)}")

        df.loc[len(df.index)] = [
            int(file.name),
            len(tp),
            len(tn),
            len(fp),
            len(fn),
        ]
        result[int(file.name)] = {}
        result[int(file.name)]["False Positive"] = [good_entities[i] for i in fp]
        result[int(file.name)]["False Negative"] = [mal_entities[i] for i in fn]

    df = calc(df)
    save_path = f"{ROOT_PATH}/baselines/baseline/shadewatcher"
    df.to_csv(f"{save_path}/result.csv")
    import json

    with open(f"{save_path}/result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    # break

    # lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

    # # 使用LOF模型对所有节点进行拟合和预测
    # # 正常节点预测结果为1，异常节点预测结果为-1
    # predictions_lof = lof_model.fit_predict(all_nodes)

    # # 打印分类报告和混淆矩阵
    # print(classification_report(labels, predictions_lof))
    # print(confusion_matrix(labels, predictions_lof))


import os

if __name__ == "__main__":
    prepare_shadewatcher()
    os.chdir(f"{ROOT_PATH}/baselines/ShadeWatcher/recommend")

    for i in range(1, SCENARIO_NUM + 1):
        os.system(f"python driver.py --dataset {i} --save_embedding")
    test_shadewatcher()
