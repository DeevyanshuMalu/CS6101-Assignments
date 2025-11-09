from scipy.sparse import coo_matrix, save_npz
from transformers import AutoModelForMaskedLM, AutoTokenizer
import argparse
import json
import os
from tqdm import tqdm
from model import *
from utils import *

with open("data/id_to_corpus_dict.json", "r") as f:
    id_to_corpus = json.load(f)


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
        "--quantize-bits",
        type=int,
        default=0,
        help="Quantize impacts to N bits when exporting",
    )
    parser.add_argument(
        "--checkpoint-to-load",
        type=str,
        default=1,
        help="Path to model checkpoint to load",
    )
    parser.add_argument(
        "--doc-expansion", action="store_true", help="Expansion handling"
    )
    args = parser.parse_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_args()

freeze_str = "_unfrozen" if args.unfreeze_bert else ""
expansion_str = "_expanded" if args.doc_expansion else ""

model_path = f"models/deepimpact_{args.epochs}ep_{args.lr}lr{freeze_str}{expansion_str}/checkpoint-{args.checkpoint_to_load}.pt"

model = DeepImpactModel(
    device=device,
    mlp_hidden=256,
    freeze_bert=(not args.unfreeze_bert),
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

save_path = (
    os.path.dirname(model_path)
    + f"/impact_scores/checkpoint-{args.checkpoint_to_load}/"
)
os.makedirs(save_path, exist_ok=True)
impact_scores_dict = compute_impact_scores(
    model,
    dict(list(id_to_corpus.items())),
    batch_size=32,
    max_length=args.max_length,
    quantize_bits=args.quantize_bits,
    doc_expansion=args.doc_expansion,
)

rows = []
cols = []
data = []

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
for i, (doc_id, impacts) in tqdm(enumerate(impact_scores_dict.items())):
    for vocab_id, impact in impacts.items():
        rows.append(vocab_id)
        cols.append(i)
        data.append(impact)

impact_scores_matrix = coo_matrix(
    (data, (rows, cols)), shape=(len(tokenizer), len(impact_scores_dict))
)
# Save sparse matrix
impact_scores_matrix = impact_scores_matrix.tocsc()
matrix_path = save_path + f"/quantize_{args.quantize_bits}_bits.npz"
save_npz(matrix_path, impact_scores_matrix)
print(f"Impact scores matrix saved to {matrix_path}")