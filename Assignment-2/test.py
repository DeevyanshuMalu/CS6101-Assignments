import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import pickle
from utils import *
from data import *

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open("data/hotpotqa_vocab_id_to_term.json", "r") as f:
    id_to_term = json.load(f)
print("Loaded id_to_term")
with open("data/hotpotqa_vocab_term_to_id.json", "r") as f:
    term_to_id = json.load(f)
print("Loaded term_to_id")
with open("data/test_doc_vectors_dict.json", "r") as f:
    doc_vectors_dict = json.load(f)
print("Loaded doc_vectors_dict")
with open("data/test_query_vectors_dict.json", "r") as f:
    query_vectors_dict = json.load(f)
print("Loaded query_vectors_dict")
with open("data/test_queries.jsonl", "r") as f:
    test_queries = [json.loads(line) for line in f]
print("Loaded test_queries")
with open("data/test_documents.jsonl", "r") as f:
    test_documents = [json.loads(line) for line in f]
print("Loaded test_documents")
with open("data/qrel_dict.json", "r") as f:
    qrel_dict = json.load(f)
print("Loaded qrel_dict")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model with custom regularization."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=300, help="Dimension of embeddings"
    )
    parser.add_argument(
        "--lambda_", type=float, default=0.1, help="Regularization strength"
    )
    parser.add_argument(
        "--task_num", type=int, default=2, help="Task number", choices=[1, 2, 3, 4]
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Margin for triplet loss (if applicable)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature for InfoNCE loss (if applicable)",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="triplet",
        help="Loss function to use",
        choices=["triplet", "infoNCE"],
    )
    parser.add_argument(
        "--test_epoch", type=int, default=10, help="Epoch number of the model to test"
    )
    parser.add_argument(
        "--doc_batch_size",
        type=int,
        default=10000,
        help="Batch size when building document tensors on-the-fly",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=100,
        help="Batch size for processing queries",
    )
    parser.add_argument(
        "--use_bert", action="store_true", help="Whether to use BERT embeddings"
    )
    return parser.parse_args()


args = parse_args()
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
embed_dim = args.embed_dim
lambda_ = args.lambda_
task_num = args.task_num
margin = args.margin
tau = args.tau
loss_fn = args.loss_fn
test_epoch = args.test_epoch
doc_batch_size = args.doc_batch_size
query_batch_size = args.query_batch_size
use_bert = args.use_bert

if task_num != 1:
    assert not use_bert, "BERT embeddings not supported for tasks 2-4."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

query_ids_to_qrel_ids = {
    text_to_id(query["question"]): query["query_id"] for query in test_queries
}
doc_ids_to_qrel_ids = {text_to_id(doc["text"]): doc["doc_id"] for doc in test_documents}

# Prepare query data
query_data = list(query_ids_to_qrel_ids.items())
doc_id_list = list(doc_ids_to_qrel_ids.keys())
num_docs = len(doc_id_list)

if task_num == 1:
    A = torch.zeros((embed_dim, len(term_to_id)), device=device)
else:
    A = torch.load(
        f"translation_matrices/task{task_num}_{loss_fn}_lambda{lambda_}_margin{margin}_tau{tau}_embed{embed_dim}/A_epoch{test_epoch}.pth"
    )
    if args.task_num == 3:
        # Load sparse indices tensor
        with open("data/wordnet_pairs.pkl", "rb") as f:
            sparse_indices = pickle.load(f)
        sparse_indices_tensor = torch.tensor(
            list(sparse_indices), device=device, dtype=torch.long
        )
        print("Loaded sparse_indices_tensor for Task 3")
    elif args.task_num == 4:
        # Load sparse indices tensor
        with open("data/glove_pairs.pkl", "rb") as f:
            sparse_indices = pickle.load(f)
        sparse_indices_tensor = torch.tensor(
            list(sparse_indices), device=device, dtype=torch.long
        )
        print("Loaded sparse_indices_tensor for Task 4")

A = A.to(device)

dcgs = []
mrrs = []
recalls = []
num_queries_to_evaluate = len(query_data)

# Process queries in batches
for query_start in tqdm(
    range(0, min(num_queries_to_evaluate, len(query_data)), query_batch_size)
):
    query_end = min(
        query_start + query_batch_size, min(num_queries_to_evaluate, len(query_data))
    )
    query_batch_data = query_data[query_start:query_end]

    # Build query batch tensor
    if not use_bert:
        query_batch_tensor = torch.zeros(
            (len(query_batch_data), len(term_to_id)), device=device
        )
        for i, (query_id, qrel_id) in enumerate(query_batch_data):
            query_vec = query_vectors_dict.get(query_id, {})
            for term in query_vec:
                if term in term_to_id:
                    query_batch_tensor[i, term_to_id[term]] = query_vec[term]
    else:
        pass  # BERT implementation would go here

    # Compute similarities for all queries in batch against all documents
    # keep the big result matrix on CPU to avoid GPU OOM
    query_batch_similarities = torch.zeros((len(query_batch_data), num_docs), device="cpu")

    for doc_start in tqdm(range(0, num_docs, doc_batch_size)):
        doc_end = min(doc_start + doc_batch_size, num_docs)
        batch_doc_ids = doc_id_list[doc_start:doc_end]

        if not use_bert:
            doc_batch = torch.zeros(
                (len(batch_doc_ids), len(term_to_id)), device=device
            )
            for k, doc_id in enumerate(batch_doc_ids):
                doc_vec = doc_vectors_dict.get(doc_id, {})
                for term in doc_vec:
                    if term in term_to_id:
                        doc_batch[k, term_to_id[term]] = doc_vec[term]
        else:
            pass  # BERT implementation would go here

        # all_sim expects (num_queries, vocab) and (num_docs_in_batch, vocab)
        if task_num in [3, 4]:
            batch_sims = all_sim_sparse(
                query_batch_tensor, doc_batch, A, lambda_, sparse_indices_tensor
            )  # shape (query_batch_size, doc_batch_size)
        else:
            batch_sims = all_sim(
                query_batch_tensor, doc_batch, A, lambda_
            )  # shape (query_batch_size, doc_batch_size)
        # move results to CPU immediately and free GPU memory
        query_batch_similarities[:, doc_start:doc_end] = batch_sims.detach().cpu()
        del batch_sims
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Process results for each query in the batch
    for i, (query_id, qrel_id) in enumerate(query_batch_data):
        similarities = query_batch_similarities[i]

        # rank and evaluate
        ranked_doc_tensor_indices = torch.argsort(similarities, descending=True)
        ranked_doc_ids = [
            doc_ids_to_qrel_ids[doc_id_list[int(idx)]]
            for idx in ranked_doc_tensor_indices
        ]

        good_doc_ids = qrel_dict[str(qrel_id)]

        dgc = DCG_at_k(good_doc_ids, ranked_doc_ids[:10])
        mrr = MRR_at_k(good_doc_ids, ranked_doc_ids[:10])
        recall = recall_at_k(good_doc_ids, ranked_doc_ids[:100])
        dcgs.append(dgc)
        mrrs.append(mrr)
        recalls.append(recall)

if task_num == 1:
    save_name = f"results/task{task_num}"
    os.makedirs(
        f"{save_name}",
        exist_ok=True,
    )
    f = open(
        f"{save_name}/results.txt",
        "w",
    )
else:
    save_name = f"results/task{task_num}_{loss_fn}_lambda{lambda_}_margin{margin}_tau{tau}_embed{embed_dim}"
    os.makedirs(
        f"{save_name}",
        exist_ok=True,
    )
    f = open(
        f"{save_name}/test_epoch{test_epoch}_results.txt",
        "w",
    )

f.write(
    f"Average DCG@10: {torch.mean(torch.tensor(dcgs))/torch.max(torch.tensor(dcgs)):.4f}\n"
)
f.write(f"Average MRR@10: {torch.mean(torch.tensor(mrrs)):.4f}\n")
f.write(f"Average Recall@100: {torch.mean(torch.tensor(recalls)):.4f}\n")

f.close()
