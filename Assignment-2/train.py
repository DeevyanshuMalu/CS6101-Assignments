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

with open('data/hotpotqa_vocab_id_to_term.json', 'r') as f:
    id_to_term = json.load(f)
print("Loaded id_to_term")
with open('data/hotpotqa_vocab_term_to_id.json', 'r') as f:
    term_to_id = json.load(f)
print("Loaded term_to_id")
with open('data/hotpotqa_bm25_vectors_dict.json', 'r') as f:
    bm25_vectors_dict = json.load(f)
print("Loaded bm25_vectors_dict")
with open('data/train_data.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]
print("Loaded train_data")
with open('data/validation_data.jsonl', 'r') as f:
    validation_data = [json.loads(line) for line in f]
print("Loaded validation_data")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with custom regularization.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--embed_dim', type=int, default=300, help='Dimension of embeddings')
    parser.add_argument('--lambda_', type=float, default=1.0, help='Regularization strength')
    parser.add_argument('--task_num', type=int, default=2, help='Task number', choices=[2, 3, 4])
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss (if applicable)')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature for InfoNCE loss (if applicable)')
    parser.add_argument('--loss_fn', type=str, default='triplet', help='Loss function to use', choices=['triplet', 'infoNCE'])
    return parser.parse_args()

def task2():
    A = nn.Parameter(torch.randn(embed_dim, vocab_size, device=device) * 0.0001)
    print(A.shape)
    
    optimizer = torch.optim.Adam([A], lr=learning_rate)

    for epoch in range(epochs):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            queries = batch['query'].to(device)
            good_docs = batch['good_doc'].to(device)
            bad_docs = batch['bad_doc'].to(device)

            queries = queries / torch.norm(queries, dim=1, keepdim=True)
            good_docs = good_docs / torch.norm(good_docs, dim=1, keepdim=True)
            bad_docs = bad_docs / torch.norm(bad_docs, dim=1, keepdim=True)

            optimizer.zero_grad()
            if loss_fn == 'triplet':
                loss = triplet_loss(queries, good_docs, bad_docs, A, lambda_, margin)
            elif loss_fn == 'infoNCE':
                loss = infoNCE_loss(queries, good_docs, A, lambda_, tau)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        with torch.no_grad():
            for val_batch in tqdm(validation_loader, desc="Validation"):
                val_queries = val_batch['query'].to(device)
                val_good_docs = val_batch['good_doc'].to(device)
                val_bad_docs = val_batch['bad_doc'].to(device)

                val_queries = val_queries / torch.norm(val_queries, dim=1, keepdim=True)
                val_good_docs = val_good_docs / torch.norm(val_good_docs, dim=1, keepdim=True)
                val_bad_docs = val_bad_docs / torch.norm(val_bad_docs, dim=1, keepdim=True)

                if loss_fn == 'triplet':
                    loss = triplet_loss(val_queries, val_good_docs, val_bad_docs, A, lambda_, margin)
                elif loss_fn == 'infoNCE':
                    loss = infoNCE_loss(val_queries, val_good_docs, A, lambda_, tau)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(validation_loader)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        os.makedirs(f'translation_matrices/task{task_num}_{loss_fn}_lambda{lambda_}_margin{margin}_tau{tau}', exist_ok=True)
        torch.save(A, f'translation_matrices/task{task_num}_{loss_fn}_lambda{lambda_}_margin{margin}_tau{tau}/A_epoch{epoch+1}.pth')

args = parse_args()

vocab_size = len(id_to_term)
embed_dim = args.embed_dim
lambda_ = args.lambda_
epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
task_num = args.task_num
margin = args.margin
tau = args.tau
loss_fn = args.loss_fn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_dataset = HotpotQADataset(train_data, term_to_id, id_to_term, bm25_vectors_dict)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = HotpotQADataset(validation_data, term_to_id, id_to_term, bm25_vectors_dict)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

wandb.init(project="CS6101 Assgn 2", config={
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "embed_dim": embed_dim,
    "lambda_": lambda_,
    "task_num": task_num,
    "margin": margin,
    "tau": tau,
    "loss_fn": loss_fn
})

if args.task_num == 2:
    task2()


# A_new = torch.load(f'translation_matrices/A_task{args.task_num}_{args.loss_fn}.pth')
# print(A_new)
# print(A_new.shape)