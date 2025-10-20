import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
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
with open("data/wikiclir_bm25_vectors_dict.json", "r") as f:
    bm25_vectors_dict = json.load(f)
print("Loaded bm25_vectors_dict")
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
use_bert = args.use_bert

if task_num != 1:
    assert not use_bert, "BERT embeddings not supported for tasks 2-4."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

query_ids_to_qrel_ids = {
    text_to_id(query["question"]): query["query_id"] for query in test_queries
}
doc_ids_to_qrel_ids = {text_to_id(doc["text"]): doc["doc_id"] for doc in test_documents}

# Don't precompute large doc_tensor or full query_tensor.
doc_id_list = list(doc_ids_to_qrel_ids.keys())
num_docs = len(doc_id_list)

if task_num == 1:
    A = torch.zeros((embed_dim, len(term_to_id)), device=device)
else:
    A = torch.load(
        f"translation_matrices/task{task_num}_{loss_fn}_lambda{lambda_}_margin{margin}_tau{tau}/A_epoch{test_epoch}.pth"
    )
A = A.to(device)

dcgs = []
mrrs = []
recalls = []
num_queries_to_evaluate = 100
for query_idx, (query_id, qrel_id) in tqdm(
    enumerate(query_ids_to_qrel_ids.items()), total=num_queries_to_evaluate
):
    if query_idx == num_queries_to_evaluate:
        break
    # print("qrel_id:", qrel_id)
    # build one-query tensor
    if not use_bert:
        query_tensor = torch.zeros((1, len(term_to_id)), device=device)
        query_vec = bm25_vectors_dict.get(query_id, {})
        for term in query_vec:
            if term in term_to_id:
                query_tensor[0, term_to_id[term]] = query_vec[term]
        query_tensor /= torch.norm(query_tensor, p=2) + 1e-8  # normalize
    else:
        pass

    # compute similarity to all docs in batches, building doc batches on-the-fly
    similarities = torch.zeros(num_docs, device=device)
    for start in range(0, num_docs, doc_batch_size):
        end = min(start + doc_batch_size, num_docs)
        batch_doc_ids = doc_id_list[start:end]

        if not use_bert:
            doc_batch = torch.zeros((len(batch_doc_ids), len(term_to_id)), device=device)
            for k, doc_id in enumerate(batch_doc_ids):
                doc_vec = bm25_vectors_dict.get(doc_id, {})
                for term in doc_vec:
                    if term in term_to_id:
                        doc_batch[k, term_to_id[term]] = doc_vec[term]

            doc_batch /= torch.norm(doc_batch, p=2, dim=1, keepdim=True) + 1e-8  # normalize
        else:
            pass

        # all_sim expects (num_queries, vocab) and (num_docs_in_batch, vocab)
        batch_sims = all_sim(
            query_tensor, doc_batch, A, lambda_
        )  # shape (1, batch_size)
        similarities[start:end] = batch_sims.squeeze(0)

    # rank and evaluate
    ranked_doc_tensor_indices = torch.argsort(similarities, descending=True)
    ranked_doc_ids = [
        doc_ids_to_qrel_ids[doc_id_list[int(idx)]] for idx in ranked_doc_tensor_indices
    ]

    good_doc_ids = qrel_dict[str(qrel_id)]

    dgc = DGC_at_k(good_doc_ids, ranked_doc_ids[:10])
    mrr = MRR_at_k(good_doc_ids, ranked_doc_ids[:10])
    recall = recall_at_k(good_doc_ids, ranked_doc_ids[:100])
    dcgs.append(dgc)
    mrrs.append(mrr)
    recalls.append(recall)

    # print(
    #     f"Query ID: {qrel_id} | DGC: {dgc:.4f} | MRR: {mrr:.4f} | Recall: {recall:.4f}"
    # )

if task_num == 1:
    save_name = (
        f"results/task{task_num}"
    )
    os.makedirs(
        f"{save_name}",
        exist_ok=True,
    )
    f = open(
        f"{save_name}/results.txt",
        "w",
    )
else:
    save_name = (
        f"results/task{task_num}_{loss_fn}_lambda{lambda_}_margin{margin}_tau{tau}"
    )
    os.makedirs(
        f"{save_name}",
        exist_ok=True,
    )
    f = open(
        f"{save_name}/test_epoch{test_epoch}_results.txt",
        "w",
    )

f.write(
    f"Average DGC@10: {torch.mean(torch.tensor(dcgs))/torch.max(torch.tensor(dcgs)):.4f}\n"
)
f.write(f"Average MRR@10: {torch.mean(torch.tensor(mrrs)):.4f}\n")
f.write(f"Average Recall@100: {torch.mean(torch.tensor(recalls)):.4f}\n")

f.close()
