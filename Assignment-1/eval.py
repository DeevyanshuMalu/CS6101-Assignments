import os
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
from datasets import load_dataset
from utils import *
import time
import argparse
import json
import functools

random.seed(42)
np.random.seed(42)

searcher = LuceneSearcher("wiki_index")
index_reader = LuceneIndexReader("wiki_index")


def flak_wct_parallel(args):
    i, val = args
    query = i["query"]
    eval_data = None
    if val:
        eval_data, _ = get_val_eval_ids(searcher, query)
    else:
        _, eval_data = get_val_eval_ids(searcher, query)
    S_ids = i["output_ids"]
    all_cands = set(S_ids).union(set(eval_data))
    all_bm25_vecs = {
        doc_id: get_bm25_vector(index_reader, doc_id) for doc_id in all_cands
    }
    flak_score = 0.0
    wct = [0.1, 0.5, 0.9]
    wct_score = [0.0, 0.0, 0.0]
    r_sum = 0.0
    for eval_doc_id in eval_data:
        r_i = get_relevance_score(index_reader, eval_doc_id, query)
        c_i = 0.0
        for s in S_ids:
            c_i = max(c_i, get_similarity_score(all_bm25_vecs, eval_doc_id, s))
        flak_i = r_i * c_i
        flak_score += flak_i
        for idx, w in enumerate(wct):
            if c_i >= w:
                wct_score[idx] += r_i
        r_sum += r_i
    for idx in range(3):
        wct_score[idx] /= r_sum

    return (query, flak_score, wct_score)


def flak_wct(dataset_name, k, task, val=False, lambd=None):
    if lambd is not None:
        assert task == 4
        file_name = f"results/task{task}_{lambd}/{dataset_name}/samples_{k}.jsonl"
    else:
        file_name = f"results/task{task}/{dataset_name}/samples_{k}.jsonl"
    with open(file_name, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        # print(data[0:5])
    print("K =", k)

    args_list = [(i, val) for i in data]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(flak_wct_parallel, args_list), total=len(args_list))
        )

    all_flaks_wcts = [
        {
            "query": query,
            "flak": score,
            "wct_1": wct[0],
            "wct_5": wct[1],
            "wct_9": wct[2],
        }
        for query, score, wct in results
    ]
    os.makedirs(f"eval_results/task{task}/{dataset_name}", exist_ok=True)
    with open(
        f"eval_results/task{task}/{dataset_name}/eval_{k}{'_val' if val else ''}{f'_lambda{lambd}' if lambd is not None else ''}.jsonl",
        "w",
    ) as f:
        for entry in all_flaks_wcts:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--file_name", type=str, default="temp.json")/
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        choices=["arguana", "kialo", "opinionqa"],
        help="Choose a dataset from arguana, kialo, opinionqa",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=20,
        required=True,
        choices=[5, 10, 20],
        help="Number of documents to select",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Task number to execute",
    )
    parser.add_argument(
        "--lambd", type=float, default=None, help="Lambda parameter for task 4"
    )
    parser.add_argument(
        "--val", action="store_true", help="Use validation set for evaluation"
    )
    args = parser.parse_args()
    flak_wct(args.dataset, args.K, args.task, args.val, args.lambd)
