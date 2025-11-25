import numpy as np
from collections import defaultdict
import argparse
import os
import json
from tqdm import tqdm


def average_precision(ranked_list, relevant_set):
    if not relevant_set:
        return 0.0

    hits = 0
    total = 0
    ap = 0.0
    for idx, d in enumerate(ranked_list, 1):
        if d in relevant_set:
            hits += 1
            ap += hits / idx
        total += 1
    ap /= len(relevant_set)
    return ap


def mean_average_precision(scores, relevance):
    ap_values = []
    for qid in scores:
        ranked = sorted(scores[qid].items(), key=lambda x: x[1], reverse=True)
        ranked_docs = [d for d, s in ranked]
        ap_values.append(average_precision(ranked_docs, relevance.get(qid, set())))
    return ap_values


def compute_map(lmb_2d, relevance):
    lambda_U = 1.0 - lmb_2d[0] - lmb_2d[1]
    if lambda_U < 0:
        return -1.0

    lambdas_3d = np.array([lmb_2d[0], lmb_2d[1], lambda_U])

    scores = defaultdict(dict)
    for qid in FT:
        for docid in FT[qid]:
            score = (
                lambdas_3d[0] * FT[qid][docid]
                + lambdas_3d[1] * FO[qid][docid]
                + lambdas_3d[2] * FU[qid][docid]
            )
            scores[qid][docid] = score
    aps = mean_average_precision(scores, relevance)
    return np.mean(aps)


def NDGC(scores, relevance, k=10):
    ndcg_values = []
    for qid in scores:
        ranked = sorted(scores[qid].items(), key=lambda x: x[1], reverse=True)
        ranked_docs = [d for d, s in ranked[:k]]

        dcg = 0.0
        for i, docid in enumerate(ranked_docs):
            if docid in relevance.get(qid, set()):
                dcg += 1 / np.log2(i + 2)

        ideal_relevant_docs = list(relevance.get(qid, set()))[:k]
        idcg = 0.0
        for i in range(len(ideal_relevant_docs)):
            idcg += 1 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_values.append(ndcg)
    return ndcg_values


def hill_climb_lambdas(relevance, step=0.05, max_iters=100):
    lambdas_2d = np.array(
        [0.8, 0.1]
    )  # [lambda_T, lambda_O], lambda_U = 1 - lambda_T - lambda_O
    # lambdas_2d = np.array([0.3, 0.3])  # [lambda_T, lambda_O], lambda_U = 1 - lambda_T - lambda_O

    best_map = compute_map(lambdas_2d, relevance)

    for _ in tqdm(range(max_iters)):
        improved = False

        for coord in range(2):
            for delta in [-step * 2, -step, step, step * 2]:
                trial = lambdas_2d.copy()
                trial[coord] += delta

                if trial[0] >= 0 and trial[1] >= 0 and (trial[0] + trial[1]) <= 1:
                    trial_map = compute_map(trial, relevance)

                    if trial_map > best_map:
                        print(
                            f"New best MAP: {best_map} with increment {trial_map - best_map} at lambdas {trial}"
                        )
                        best_map = trial_map
                        lambdas_2d = trial
                        improved = True
                        break

            if improved:
                break

        if not improved:
            break

    lambda_U = 1.0 - lambdas_2d[0] - lambdas_2d[1]
    final_lambdas = np.array([lambdas_2d[0], lambdas_2d[1], lambda_U])

    return final_lambdas, best_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DeepImpact-style model on triples (query, good, bad)."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--train-val-split", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument(
        "--unfreeze-bert",
        action="store_true",
        help="Fine-tune BERT weights during training",
    )
    parser.add_argument(
        "--quantize-bits",
        type=int,
        default=0,
        help="Quantize impacts to N bits when exporting",
    )
    parser.add_argument(
        "--checkpoint-to-load",
        type=str,
        default=2,
        help="Path to model checkpoint to load",
    )
    parser.add_argument(
        "--doc-expansion", action="store_true", help="Expansion handling"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Window size for non-consecutive pairs",
    )
    parser.add_argument(
        "--stage_1_baseline", action="store_true", help="Stage 1 baseline without reranking"
    )
    args = parser.parse_args()
    return args


args = parse_args()

with open("data/id_to_query_dict.json", "r") as f:
    id_to_query = json.load(f)
with open("data/train_qrels_dict.json", "r") as f:
    train_qrels = json.load(f)
with open("data/test_qrels_dict.json", "r") as f:
    test_qrels = json.load(f)
with open("data/id_to_corpus_dict.json", "r") as f:
    id_to_corpus = json.load(f)


freeze_str = "_unfrozen" if args.unfreeze_bert else ""
expansion_str = "_expanded" if args.doc_expansion else ""

model_path = f"models/deepimpact_{args.epochs}ep_{args.lr}lr{freeze_str}{expansion_str}/checkpoint-{args.checkpoint_to_load}.pt"
save_path = (
    os.path.dirname(model_path)
    + f"/precomputed_features/checkpoint-{args.checkpoint_to_load}/"
)

with open(
    save_path
    + f"/FT_train_{args.window_size}win_quantize_{args.quantize_bits}_bits.json",
    "r",
) as f:
    FT = json.load(f)
with open(
    save_path
    + f"/FO_train_{args.window_size}win_quantize_{args.quantize_bits}_bits.json",
    "r",
) as f:
    FO = json.load(f)
with open(
    save_path
    + f"/FU_train_{args.window_size}win_quantize_{args.quantize_bits}_bits.json",
    "r",
) as f:
    FU = json.load(f)

if not args.stage_1_baseline:
    lambdas, best_map = hill_climb_lambdas(train_qrels)

    print("Optimal Lambdas:", lambdas)
    print("Best MAP:", best_map)

with open(
    save_path
    + f"/FT_test_{args.window_size}win_quantize_{args.quantize_bits}_bits.json",
    "r",
) as f:
    FT_test = json.load(f)
with open(
    save_path
    + f"/FO_test_{args.window_size}win_quantize_{args.quantize_bits}_bits.json",
    "r",
) as f:
    FO_test = json.load(f)
with open(
    save_path
    + f"/FU_test_{args.window_size}win_quantize_{args.quantize_bits}_bits.json",
    "r",
) as f:
    FU_test = json.load(f)

if not args.stage_1_baseline:
    scores = defaultdict(dict)
    for qid in FT_test:
        for docid in FT_test[qid]:
            score = (
                lambdas[0] * FT_test[qid][docid]
                + lambdas[1] * FO_test[qid][docid]
                + lambdas[2] * FU_test[qid][docid]
            )
            scores[qid][docid] = score

    test_map = mean_average_precision(scores, test_qrels)
    map_quora = np.mean(test_map[:-8])
    map_trec = np.mean(test_map[-8:])
    print("Test MAP (Quora) with optimal lambdas:", map_quora)
    print("Test MAP (Trec) with optimal lambdas:", map_trec)

    test_ndcg = NDGC(scores, test_qrels, k=10)
    ndcg_quora = np.mean(test_ndcg[:-8])
    ndcg_trec = np.mean(test_ndcg[-8:])
    print("Test NDCG@10 (Quora) with optimal lambdas:", ndcg_quora)
    print("Test NDCG@10 (Trec) with optimal lambdas:", ndcg_trec)

    new_save_path = os.path.dirname(model_path) + f"/results/checkpoint-{args.checkpoint_to_load}/"
    os.makedirs(new_save_path, exist_ok=True)
    with open(new_save_path + f"/results_{args.window_size}win_quantize_{args.quantize_bits}_bits.json", "w") as f:
        json.dump({
            "lambdas": lambdas.tolist(),
            "MAP_Quora": map_quora,
            "MAP_Trec": map_trec,
            "NDCG_Quora": ndcg_quora,
            "NDCG_Trec": ndcg_trec
        }, f, indent=4)
else:
    orig_map = defaultdict(dict)
    for qid in test_qrels:
        for i, docid in enumerate(FT_test[qid]):
            # orig_map[qid][docid] = len(FT_test[qid]) - i
            orig_map[qid][docid] = i
    # print(orig_map)
    
    test_map = mean_average_precision(orig_map, test_qrels)
    map_quora = np.mean(test_map[:-8])
    map_trec = np.mean(test_map[-8:])
    print("Test MAP (Quora) without reranking:", map_quora)
    print("Test MAP (Trec) without reranking:", map_trec)

    test_ndcg = NDGC(orig_map, test_qrels, k=10)
    ndcg_quora = np.mean(test_ndcg[:-8])
    ndcg_trec = np.mean(test_ndcg[-8:])
    print("Test NDCG@10 (Quora) without reranking:", ndcg_quora)
    print("Test NDCG@10 (Trec) without reranking:", ndcg_trec)

    new_save_path = os.path.dirname(model_path) + f"/results/checkpoint-{args.checkpoint_to_load}/"
    os.makedirs(new_save_path, exist_ok=True)
    with open(new_save_path + f"/results_no_reranking_quantize_{args.quantize_bits}_bits.json", "w") as f:
        json.dump({
            "MAP_Quora": map_quora,
            "MAP_Trec": map_trec,
            "NDCG_Quora": ndcg_quora,
            "NDCG_Trec": ndcg_trec
        }, f, indent=4)