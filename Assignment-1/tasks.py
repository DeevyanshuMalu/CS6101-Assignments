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
import gurobipy as gp
from gurobipy import GRB
import json

searcher = LuceneSearcher("wiki_index")
index_reader = LuceneIndexReader("wiki_index")


def task1(row):
    query = row["question"]
    candidates_ids = get_candidate_ids(searcher, query)
    all_bm25_vecs = {
        doc_id: get_bm25_vector(index_reader, doc_id) for doc_id in candidates_ids
    }
    relevance_scores_dict = {
        doc_id: get_relevance_score(index_reader, doc_id, query)
        for doc_id in candidates_ids
    }
    similarity_matrix = {
        (doc_id_i, doc_id_j): get_similarity_score(
            index_reader, all_bm25_vecs, doc_id_i, doc_id_j
        )
        for doc_id_i in candidates_ids
        for doc_id_j in candidates_ids
        if doc_id_i <= doc_id_j
    }
    S = []
    c_is = {doc_id: 0.0 for doc_id in candidates_ids}
    score = 0.0
    while len(S) < K:
        best_doc_id = ""
        best_change = 0.0
        best_c_is = c_is.copy()
        for doc_id in candidates_ids:
            if doc_id in S:
                continue
            score_new = 0.0
            c_is_new = c_is.copy()
            for doc_id2 in candidates_ids:
                c_i = get_c(doc_id2, S + [doc_id], c_is, similarity_matrix, fast)
                score_new += relevance_scores_dict[doc_id2] * c_i
                c_is_new[doc_id2] = c_i
            change = score_new - score
            if change >= best_change:
                best_change = change
                best_doc_id = doc_id
                best_c_is = c_is_new
        score += best_change
        c_is = best_c_is
        S.append(best_doc_id)
    return S


def task2(row):
    query = row["question"]
    candidates_ids = get_candidate_ids(searcher, query)
    all_bm25_vecs = {
        doc_id: get_bm25_vector(index_reader, doc_id) for doc_id in candidates_ids
    }
    relevance_scores_dict = {
        doc_id: get_relevance_score(index_reader, doc_id, query)
        for doc_id in candidates_ids
    }
    similarity_matrix = {
        (doc_id_i, doc_id_j): get_similarity_score(
            index_reader, all_bm25_vecs, doc_id_i, doc_id_j
        )
        for doc_id_i in candidates_ids
        for doc_id_j in candidates_ids
        if doc_id_i <= doc_id_j
    }

    model = gp.Model(env=env)
    model.Params.OutputFlag = 0
    x = model.addVars(len(candidates_ids), vtype=GRB.BINARY, name="x")
    z = model.addVars(
        len(candidates_ids), len(candidates_ids), vtype=GRB.BINARY, name="z"
    )

    model.addConstrs(
        (
            z[i, j] <= x[i]
            for i in range(len(candidates_ids))
            for j in range(len(candidates_ids))
        ),
        "z<x",
    )
    model.addConstr(gp.quicksum(x[i] for i in range(len(candidates_ids))) == K, "x_sum")
    model.setObjective(
        gp.quicksum(
            relevance_scores_dict[doc_id_i]
            * similarity_matrix[(min(doc_id_i, doc_id_j), max(doc_id_i, doc_id_j))]
            * z[i, j]
            for i, doc_id_i in enumerate(candidates_ids)
            for j, doc_id_j in enumerate(candidates_ids)
        ),
        GRB.MAXIMIZE,
    )

    # model.Params.TimeLimit = 10.0

    model.optimize()

    return [candidates_ids[i] for i in range(len(candidates_ids)) if x[i].X == 1]


def task3(row):
    query = row["question"]
    candidates_ids = get_candidate_ids(searcher, query)
    all_bm25_vecs = {
        doc_id: get_bm25_vector(index_reader, doc_id) for doc_id in candidates_ids
    }
    relevance_scores_dict = {
        doc_id: get_relevance_score(index_reader, doc_id, query)
        for doc_id in candidates_ids
    }
    similarity_matrix = {
        (doc_id_i, doc_id_j): get_similarity_score(
            index_reader, all_bm25_vecs, doc_id_i, doc_id_j
        )
        for doc_id_i in candidates_ids
        for doc_id_j in candidates_ids
        if doc_id_i <= doc_id_j
    }

    model = gp.Model(env=env)
    model.Params.OutputFlag = 0
    x = model.addVars(len(candidates_ids), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
    z = model.addVars(
        len(candidates_ids),
        len(candidates_ids),
        lb=0,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="z",
    )

    model.addConstrs(
        (
            z[i, j] <= x[i]
            for i in range(len(candidates_ids))
            for j in range(len(candidates_ids))
        ),
        "z<x",
    )
    model.addConstr(gp.quicksum(x[i] for i in range(len(candidates_ids))) == K, "x_sum")
    model.setObjective(
        gp.quicksum(
            relevance_scores_dict[doc_id_i]
            * similarity_matrix[(min(doc_id_i, doc_id_j), max(doc_id_i, doc_id_j))]
            * z[i, j]
            for i, doc_id_i in enumerate(candidates_ids)
            for j, doc_id_j in enumerate(candidates_ids)
        ),
        GRB.MAXIMIZE,
    )

    # model.Params.TimeLimit = 10.0

    model.optimize()

    x_probs = [x[i].X for i in range(len(candidates_ids))]
    # print(x_probs)

    output_ids = [
        doc_id for doc_id, p in zip(candidates_ids, x_probs) if random.random() < p
    ]
    return output_ids


def task4(row):
    query = row["question"]
    candidates_ids = get_candidate_ids(searcher, query)
    all_bm25_vecs = {
        doc_id: get_bm25_vector(index_reader, doc_id) for doc_id in candidates_ids
    }
    relevance_scores_dict = {
        doc_id: get_relevance_score(index_reader, doc_id, query)
        for doc_id in candidates_ids
    }
    similarity_matrix = {
        (doc_id_i, doc_id_j): get_similarity_score(
            index_reader, all_bm25_vecs, doc_id_i, doc_id_j
        )
        for doc_id_i in candidates_ids
        for doc_id_j in candidates_ids
        if doc_id_i <= doc_id_j
    }

    S = []
    c_is = {doc_id: 0.0 for doc_id in candidates_ids}
    while len(S) < K:
        best_doc_id = ""
        best_score = 0.0
        for doc_id in candidates_ids:
            if doc_id in S:
                continue
            c_i = c_is[doc_id]
            score_new = relevance_scores_dict[doc_id] * lambd - c_i * (1 - lambd)
            if score_new >= best_score:
                best_score = score_new
                best_doc_id = doc_id
        for doc_id in candidates_ids:
            c_is[doc_id] = get_c(
                doc_id, S + [best_doc_id], c_is, similarity_matrix, fast
            )
        S.append(best_doc_id)

    return S


def task5(row):
    query = row["question"]
    raise NotImplementedError("Task 5 is not implemented yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        choices=["arguana", "kialo", "opinionqa"],
        help="Choose a dataset from arguana, kialo, opinionqa",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use fast mode for c(i) computation"
    )
    parser.add_argument(
        "--K", type=int, default=20, help="Number of documents to select"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Task number to execute",
    )
    parser.add_argument(
        "--lambd", type=float, default=0.5, help="Lambda parameter for task 4"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "arguana":
        ds = load_dataset("timchen0618/Arguana", split="test")
    elif args.dataset == "kialo":
        ds = load_dataset("timchen0618/Kialo", split="test")
    elif args.dataset == "opinionqa":
        ds = load_dataset("timchen0618/OpinionQA", split="test")

    if args.task == 1:
        task = task1
    elif args.task == 2:
        env = gp.Env()
        task = task2
    elif args.task == 3:
        env = gp.Env()
        task = task3
    elif args.task == 4:
        task = task4
    elif args.task == 5:
        task = task5

    K = args.K
    fast = args.fast
    lambd = args.lambd

    file = open(f"task{args.task}/samples_{args.dataset}.jsonl", "w")
    os.makedirs(f"task{args.task}", exist_ok=True)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(task, ds), total=len(ds)))
    for S, row in zip(results, ds):
        out_dict = {"query": row["question"], "output_ids": S}
        file.write(f"{json.dumps(out_dict)}\n")
