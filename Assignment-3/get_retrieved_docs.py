from scipy.sparse import coo_matrix, load_npz
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import argparse
import json
import os
import pickle

with open("data/id_to_corpus_dict.json", "r") as f:
    id_to_corpus = json.load(f)
with open("data/id_to_query_dict.json", "r") as f:
    id_to_query = json.load(f)
with open("data/test_qrels_dict.json", "r") as f:
    test_qrels = json.load(f)
with open("data/train_qrels_dict.json", "r") as f:
    train_qrels = json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DeepImpact-style model on triples (query, good, bad)."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
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
    args = parser.parse_args()
    return args


args = parse_args()

corpus_ids = list(id_to_corpus.keys())
test_query_ids = list(test_qrels.keys())
train_query_ids = list(train_qrels.keys())

freeze_str = "_unfrozen" if args.unfreeze_bert else ""
expansion_str = "_expanded" if args.doc_expansion else ""

model_path = f"models/deepimpact_{args.epochs}ep_{args.lr}lr{freeze_str}{expansion_str}/checkpoint-{args.checkpoint_to_load}.pt"
save_path = (
    os.path.dirname(model_path)
    + f"/impact_scores/checkpoint-{args.checkpoint_to_load}/"
)
matrix_path = save_path + f"/quantize_{args.quantize_bits}_bits.npz"
impact_scores_matrix = load_npz(matrix_path)

rows = []
cols = []
data = []

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for i, q_id in tqdm(enumerate(test_qrels)):
    query = id_to_query[q_id]
    tokenized_query = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    unique_vocab_ids = set(tokenized_query["input_ids"])
    for vocab_id in unique_vocab_ids:
        rows.append(vocab_id)
        cols.append(i)
        data.append(1)

query_term_matrix = coo_matrix(
    (data, (rows, cols)),
    shape=(impact_scores_matrix.shape[0], len(test_qrels)),
    dtype=impact_scores_matrix.dtype,
)
query_term_matrix = query_term_matrix.tocsc()

query_doc_scores = query_term_matrix.T @ impact_scores_matrix
print(query_doc_scores.shape)  # (num_queries, num_docs)

query_doc_scores = query_doc_scores.toarray()
topk_doc_ids = np.argsort(query_doc_scores, axis=1)[:, -100:] # (num_queries, 100)
topk_doc_ids = topk_doc_ids[:, ::-1]  # sort topk doc ids in descending order

query_to_retrieved_doc_ids = {}
for i in tqdm(range(len(test_qrels))):
    rel_docs = test_qrels[test_query_ids[i]]
    doc_mat_ids = topk_doc_ids[i]
    retrieved_doc_ids = [corpus_ids[doc_mat_id] for doc_mat_id in doc_mat_ids]
    query_to_retrieved_doc_ids[test_query_ids[i]] = retrieved_doc_ids

results_path = os.path.dirname(model_path) + f"/query_to_retrieved_doc_ids/checkpoint-{args.checkpoint_to_load}/"
os.makedirs(results_path, exist_ok=True)
with open(results_path + f"/test_quantize_{args.quantize_bits}_bits.json", "w") as f:
    json.dump(query_to_retrieved_doc_ids, f, indent=4)



rows = []
cols = []
data = []

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for i, q_id in tqdm(enumerate(train_qrels)):
    query = id_to_query[q_id]
    tokenized_query = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    unique_vocab_ids = set(tokenized_query["input_ids"])
    for vocab_id in unique_vocab_ids:
        rows.append(vocab_id)
        cols.append(i)
        data.append(1)

query_term_matrix = coo_matrix(
    (data, (rows, cols)),
    shape=(impact_scores_matrix.shape[0], len(train_qrels)),
    dtype=impact_scores_matrix.dtype,
)
query_term_matrix = query_term_matrix.tocsc()

query_doc_scores = query_term_matrix.T @ impact_scores_matrix
print(query_doc_scores.shape)  # (num_queries, num_docs)

query_doc_scores = query_doc_scores.toarray()
topk_doc_ids = np.argsort(query_doc_scores, axis=1)[:, -100:] # (num_queries, 100)
topk_doc_ids = topk_doc_ids[:, ::-1]  # sort topk doc ids in descending order

query_to_retrieved_doc_ids = {}
for i in tqdm(range(len(train_qrels))):
    rel_docs = train_qrels[train_query_ids[i]]
    doc_mat_ids = topk_doc_ids[i]
    retrieved_doc_ids = [corpus_ids[doc_mat_id] for doc_mat_id in doc_mat_ids]
    query_to_retrieved_doc_ids[train_query_ids[i]] = retrieved_doc_ids

results_path = os.path.dirname(model_path) + f"/query_to_retrieved_doc_ids/checkpoint-{args.checkpoint_to_load}/"
os.makedirs(results_path, exist_ok=True)
with open(results_path + f"/train_quantize_{args.quantize_bits}_bits.json", "w") as f:
    json.dump(query_to_retrieved_doc_ids, f, indent=4)