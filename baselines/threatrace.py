from baseline import *


def threatace(
    scenario=1,
    folder=f"{ROOT_PATH}/baselines/baseline/threatace",
    process_only=False,
):
    import torch
    import numpy as np
    from threaTrace_torch_test import (
        nxGraph2Loder,
        train_single_model,
        evaluation,
        no2nid_map,
        node_map,
    )

    wrong_node = {}
    g, mal_node_ids = load_all_file(scenario)
    mal_node_ids = keep_process(mal_node_ids, process_only)
    train_loaders = []
    train_loaders.append(nxGraph2Loder(g, mal_node_ids, graph_id=1))
    test_loader = nxGraph2Loder(g, mal_node_ids, is_train=False, graph_id=1)
    model_pth = f"{folder}/model_scenario{scenario}.pt"
    m = train_single_model(train_loaders, test_loader, model_pth, dont_train=True)
    m.load_state_dict(torch.load(model_pth))

    def eva(m, valid_dataloader):
        m.eval()
        predictions = []
        labels = []
        ben_mask = []
        for input_nodes, output_nodes, mfgs in valid_dataloader:
            inputs = mfgs[0].srcdata["feat"]
            edge_weights = [mfgs[0].edata["weight"], mfgs[1].edata["weight"]]
            # ben_mask.append(mfgs[0].srcdata["ben_mask"].cpu().numpy())
            ben_mask.append(mfgs[-1].dstdata["ben_mask"].cpu().numpy())
            labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
            predictions.append(m(mfgs, inputs, edge_weights).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        ben_mask = np.concatenate(ben_mask)
        TP, TN, FP, FN = evaluation(predictions, labels, ben_mask)
        # ben_mask mal:false
        tn = (predictions == labels) * ben_mask
        tp = (predictions != labels) * (1 - ben_mask)
        fp_id = np.where(np.logical_and(ben_mask == True, tn == False))
        fn_id = np.where(
            np.logical_and(ben_mask == False, tp == False)
        )  # (1 - ben_mask) - tp
        # fp_id = np.where(fp == 1)
        # fn_id = np.where(fn == 1)
        return TP, TN, FP, FN, fp_id, fn_id

    df = pd.DataFrame(columns=["scenario", "TP", "TN", "FP", "FN"])
    for i in range(1, SCENARIO_NUM + 1):
        # if i == scenario:
        #     continue
        g2, mal_node_ids2 = load_all_file(i)
        if process_only:
            mal_node_ids2 = keep_process(mal_node_ids2, process_only)

        test_loader = nxGraph2Loder(g2, mal_node_ids2, is_train=False, graph_id=i)
        TP, TN, FP, FN, fp_id, fn_id = eva(m, test_loader)
        fp_node = [no2nid_map[i][j] for j in fp_id[0]]
        fn_node = [no2nid_map[i][j] for j in fn_id[0]]
        fp_node = [no2nid_map[i][j] for j in fp_id[0] if ".exe_" in no2nid_map[i][j]]
        fn_node = [no2nid_map[i][j] for j in fn_id[0] if ".exe_" in no2nid_map[i][j]]

        wrong_node[f"scenario{i}"] = {}
        wrong_node[f"scenario{i}"]["False Positives"] = fp_node
        wrong_node[f"scenario{i}"]["False Negatives"] = fn_node
        mal_process_num = len([n for n in g2.nodes if ".exe_" in n])

        df.loc[len(df.index)] = [
            i,
            TP,
            mal_process_num - TP - len(fp_node) - len(fn_node),
            len(fp_node),  # FP,
            len(fn_node),  # FN,
        ]  # [i, node_map[i][0], node_map[i][1], TP, TN, FP, FN]
    df = calc(df)
    df.to_csv(f"{folder}/result.csv")
    import json

    with open(f"{folder}/wrong_node.json", "w", encoding="utf-8") as f:
        json.dump(wrong_node, f, ensure_ascii=False)


if __name__ == "__main__":
    threatace()
