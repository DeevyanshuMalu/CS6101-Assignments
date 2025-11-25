import argparse
import os
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import json
import pickle
from model import *
from dataset import *
from utils import *

with open("data/id_to_corpus_dict.json", "r") as f:
    id_to_corpus = json.load(f)
with open("data/id_to_query_dict.json", "r") as f:
    id_to_query = json.load(f)
with open("data/training_data.pkl", "rb") as f:
    training_data = pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepImpact-style model on triples (query, good, bad)."
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
        "--doc-expansion", action="store_true", help="Expansion handling"
    )
    args = parser.parse_args()
    return args


args = parse_args()

wandb.init(
    project="CS6101 Assgn 3",
    config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_val_split": args.train_val_split,
        "max_length": args.max_length,
        "unfreeze_bert": args.unfreeze_bert,
        "doc_expansion": args.doc_expansion,
    },
)

freeze_str = "_unfrozen" if args.unfreeze_bert else ""
expansion_str = "_expanded" if args.doc_expansion else ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add these optimizations at the beginning
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

dataset = TriplesDataset(
    training_data,
    id_to_query,
    id_to_corpus,
    args.doc_expansion,
    device,
    args.max_length,
)
train_data = dataset[: int(len(dataset) * args.train_val_split)]
val_data = dataset[int(len(dataset) * args.train_val_split) :]

train_data_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x
)
val_data_loader = DataLoader(
    val_data, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x
)

model = DeepImpactModel(
    device=device,
    mlp_hidden=256,
    freeze_bert=(not args.unfreeze_bert),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0
    pbar_train = tqdm(train_data_loader, desc="Training", leave=False)
    for batch in pbar_train:
        optimizer.zero_grad()

        # Extract all queries, pos_docs, neg_docs from batch
        queries = [item[0] for item in batch]
        pos_docs = [item[1] for item in batch]
        neg_docs = [item[2] for item in batch]

        # Batch tokenize all texts at once
        q_batch = model.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        pos_batch = model.tokenizer(
            pos_docs,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        neg_batch = model.tokenizer(
            neg_docs,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        pos_impacts = model.get_token_impacts(
            pos_batch["input_ids"], pos_batch["attention_mask"]
        )
        neg_impacts = model.get_token_impacts(
            neg_batch["input_ids"], neg_batch["attention_mask"]
        )

        pos_scores = model.get_query_doc_scores(q_batch, pos_batch, pos_impacts)
        neg_scores = model.get_query_doc_scores(q_batch, neg_batch, neg_impacts)

        loss = softmax_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * len(batch)
        pbar_train.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_data_loader.dataset)
    print(f"Epoch {epoch}/{args.epochs} - train avg loss: {avg_train_loss:.6f}")

    pbar_val = tqdm(val_data_loader, desc="Validation", leave=False)
    model.eval()
    with torch.no_grad():
        for batch in pbar_val:
            # Extract all queries, pos_docs, neg_docs from batch
            queries = [item[0] for item in batch]
            pos_docs = [item[1] for item in batch]
            neg_docs = [item[2] for item in batch]

            # Batch tokenize all texts at once
            q_batch = model.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            pos_batch = model.tokenizer(
                pos_docs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            neg_batch = model.tokenizer(
                neg_docs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)

            pos_impacts = model.get_token_impacts(
                pos_batch["input_ids"], pos_batch["attention_mask"]
            )
            neg_impacts = model.get_token_impacts(
                neg_batch["input_ids"], neg_batch["attention_mask"]
            )

            pos_scores = model.get_query_doc_scores(q_batch, pos_batch, pos_impacts)
            neg_scores = model.get_query_doc_scores(q_batch, neg_batch, neg_impacts)

            loss = softmax_loss(pos_scores, neg_scores)
            total_val_loss += loss.item() * len(batch)
            pbar_val.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_data_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} - val avg loss: {avg_val_loss:.6f}")

    wandb.log(
        {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
    )

    model_path = (
        f"models/deepimpact_{args.epochs}ep_{args.lr}lr{freeze_str}{expansion_str}"
    )
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), f"{model_path}/checkpoint-{epoch}.pt")
    print(f"Saved checkpoint: {model_path}")

wandb.finish()
