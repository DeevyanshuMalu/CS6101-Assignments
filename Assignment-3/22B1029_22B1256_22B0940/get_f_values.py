import numpy as np
from collections import defaultdict
import argparse
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import json
from tqdm import tqdm
import os

def scaled_cosine_batch(a, b):
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return (1 + cos_sim) / 2.0

def get_consecutive_pairs(seq_len):
    return [(i, i+1) for i in range(seq_len-1)]

def get_nonconsecutive_pairs(seq_len, window_size=None):
    pairs = []
    for i in range(seq_len):
        for j in range(i+2, seq_len):
            if window_size is None or (j - i) <= window_size:
                pairs.append((i, j))
    return pairs

def precompute_features_batch(
    query_embs,
    documents_embs,
    doc_ids,
    window_size=None,
    device='cpu'
):
    query_embs = torch.tensor(query_embs, device=device)
    documents_embs = torch.tensor(documents_embs, device=device)
    
    seq_len = query_embs.shape[0]
    num_docs = documents_embs.shape[0]
    
    ft_scores = torch.sum(scaled_cosine_batch(
        query_embs.unsqueeze(1),  # (seq_len, 1, hidden_size)
        documents_embs.unsqueeze(0)  # (1, num_docs, hidden_size)
    ), dim=0)  # (num_docs,)
    
    cons_pairs = get_consecutive_pairs(seq_len)
    unord_pairs = get_nonconsecutive_pairs(seq_len, window_size)
    
    fo_scores = torch.zeros(num_docs, device=device)
    cons_pairs_tensor = torch.tensor(cons_pairs, device=device)
    pair_embs = (query_embs[cons_pairs_tensor[:, 0]] + query_embs[cons_pairs_tensor[:, 1]]) / 2.0
    fo_batch = scaled_cosine_batch(
        pair_embs.unsqueeze(1),  # (num_pairs, 1, hidden_size)
        documents_embs.unsqueeze(0)  # (1, num_docs, hidden_size)
    )
    fo_scores = torch.sum(fo_batch, dim=0)
    
    fu_scores = torch.zeros(num_docs, device=device)
    unord_pairs_tensor = torch.tensor(unord_pairs, device=device)
    unord_pair_embs = (query_embs[unord_pairs_tensor[:, 0]] + query_embs[unord_pairs_tensor[:, 1]]) / 2.0
    fu_batch = scaled_cosine_batch(
        unord_pair_embs.unsqueeze(1),  # (num_pairs, 1, hidden_size)
        documents_embs.unsqueeze(0)  # (1, num_docs, hidden_size)
    )
    fu_scores = torch.sum(fu_batch, dim=0)
    
    FT = {doc_id: ft_scores[i].item() for i, doc_id in enumerate(doc_ids)}
    FO = {doc_id: fo_scores[i].item() for i, doc_id in enumerate(doc_ids)}
    FU = {doc_id: fu_scores[i].item() for i, doc_id in enumerate(doc_ids)}
    
    return FT, FO, FU

def process_batch_documents(doc_texts, tokenizer, model, max_length, device):
    if not doc_texts:
        return []
    
    docs_tokens = tokenizer(
        doc_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    docs_tokens = {k: v.to(device) for k, v in docs_tokens.items()}
    
    with torch.no_grad():
        doc_outputs = model(**docs_tokens, output_hidden_states=True)
        
        attention_mask = docs_tokens['attention_mask']  # (batch_size, seq_len)
        hidden_states = doc_outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        
        masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)  # (batch_size, seq_len, hidden_size)
        
        sum_embeddings = masked_hidden_states.sum(dim=1)  # (batch_size, hidden_size)
        seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        doc_embeddings = sum_embeddings / seq_lengths  # (batch_size, hidden_size)
    
    return doc_embeddings.cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DeepImpact-style model on triples (query, good, bad)."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--train-val-split", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--unfreeze-bert", action="store_true", help="Fine-tune BERT weights during training")
    parser.add_argument("--quantize-bits", type=int, default=0, help="Quantize impacts to N bits when exporting")
    parser.add_argument("--checkpoint-to-load", type=str, default=2, help="Path to model checkpoint to load")
    parser.add_argument("--doc-expansion", action="store_true", help="Expansion handling")
    parser.add_argument("--window-size", type=int, default=5, help="Window size for non-consecutive pairs")
    args = parser.parse_args()
    return args

def process_queries_batch(query_data, tokenizer, model, id_to_corpus, args, device):
    FT_results = defaultdict(dict)
    FO_results = defaultdict(dict)
    FU_results = defaultdict(dict)
    
    for qid in tqdm(query_data, desc="Processing queries"):
        query_text = id_to_query[qid]
        doc_ids = query_data[qid]
        doc_texts = [id_to_corpus[docid] for docid in doc_ids]
        
        query_tokens = tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
        query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
        
        with torch.no_grad():
            query_outputs = model(**query_tokens, output_hidden_states=True)
            query_embeddings = query_outputs.hidden_states[-1].squeeze().cpu().numpy()
        
        batch_size = args.batch_size
        doc_embeddings = []
        
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i + batch_size]
            batch_embeddings = process_batch_documents(
                batch_texts, tokenizer, model, args.max_length, device
            )
            doc_embeddings.extend(batch_embeddings)
        
        doc_embeddings = np.array(doc_embeddings)
        
        FT, FO, FU = precompute_features_batch(
            query_embeddings,
            doc_embeddings,
            doc_ids,
            window_size=args.window_size,
            device=device
        )
        
        FT_results[qid] = FT
        FO_results[qid] = FO
        FU_results[qid] = FU
    
    return FT_results, FO_results, FU_results


args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
model.eval()


with open("data/id_to_corpus_dict.json", "r") as f:
    id_to_corpus = json.load(f)
with open("data/id_to_query_dict.json", "r") as f:
    id_to_query = json.load(f)

freeze_str = "_unfrozen" if args.unfreeze_bert else ""
expansion_str = "_expanded" if args.doc_expansion else ""

model_path = f"models/deepimpact_{args.epochs}ep_{args.lr}lr{freeze_str}{expansion_str}/checkpoint-{args.checkpoint_to_load}.pt"
results_path = os.path.dirname(model_path) + f"/query_to_retrieved_doc_ids/checkpoint-{args.checkpoint_to_load}/"

with open(results_path + f"/test_quantize_{args.quantize_bits}_bits.json", "r") as f:
    query_to_retrieved_doc_ids_test = json.load(f)
with open(results_path + f"/train_quantize_{args.quantize_bits}_bits.json", "r") as f:
    query_to_retrieved_doc_ids_train = json.load(f)

FT_train, FO_train, FU_train = process_queries_batch(
    query_to_retrieved_doc_ids_train, tokenizer, model, id_to_corpus, args, device
)

FT_test, FO_test, FU_test = process_queries_batch(
    query_to_retrieved_doc_ids_test, tokenizer, model, id_to_corpus, args, device
)

save_path = os.path.dirname(model_path) + f"/precomputed_features/checkpoint-{args.checkpoint_to_load}/"
os.makedirs(save_path, exist_ok=True)

with open(save_path + f"/FT_train_{args.window_size}win_quantize_{args.quantize_bits}_bits.json", "w") as f:
    json.dump(FT_train, f, indent=4)
with open(save_path + f"/FO_train_{args.window_size}win_quantize_{args.quantize_bits}_bits.json", "w") as f:
    json.dump(FO_train, f, indent=4)
with open(save_path + f"/FU_train_{args.window_size}win_quantize_{args.quantize_bits}_bits.json", "w") as f:
    json.dump(FU_train, f, indent=4)
with open(save_path + f"/FT_test_{args.window_size}win_quantize_{args.quantize_bits}_bits.json", "w") as f:
    json.dump(FT_test, f, indent=4)
with open(save_path + f"/FO_test_{args.window_size}win_quantize_{args.quantize_bits}_bits.json", "w") as f:
    json.dump(FO_test, f, indent=4)
with open(save_path + f"/FU_test_{args.window_size}win_quantize_{args.quantize_bits}_bits.json", "w") as f:
    json.dump(FU_test, f, indent=4)
