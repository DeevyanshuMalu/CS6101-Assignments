import torch
import math
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

def quantize_scores(scores, bits=8):
    if bits <= 0:
        raise ValueError("bits must be > 0")
    qmax = (1 << bits) - 1
    # map real scores to [1, qmax] linearly based on min/max in the tensor.
    smin = float(scores.min().item()) if scores.numel() > 0 else 0.0
    smax = float(scores.max().item()) if scores.numel() > 0 else smin + 1.0
    if math.isclose(smax, smin):
        midpoint = (qmax + 1) // 2
        return torch.full_like(scores, midpoint, dtype=torch.long)
    scaled = (scores - smin) / (smax - smin)  # 0..1
    ints = (scaled * (qmax - 1) + 1).round().clamp(1, qmax).to(torch.long)
    return ints


def softmax_loss(pos_score, neg_score):
    exp_pos = torch.exp(pos_score)
    exp_neg = torch.exp(neg_score)
    loss = -torch.log(exp_pos / (exp_pos + exp_neg)).mean()
    return loss


def compute_impact_scores(
    model, id_to_corpus, batch_size=32, max_length=160, quantize_bits=0, doc_expansion=False
):
    model.eval()
    all_results = {}

    ids = [id for id in id_to_corpus.keys()]
    texts = [id_to_corpus[id] for id in ids]

    if doc_expansion:
        tokenizer_exp = T5Tokenizer.from_pretrained(
            "castorini/doc2query-t5-base-msmarco"
        )
        model_exp = T5ForConditionalGeneration.from_pretrained(
            "castorini/doc2query-t5-base-msmarco"
        ).to(model.device)
        model_exp.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing impact scores"):
            batch_texts = texts[i : i + batch_size]

            if doc_expansion:
                batch_tokenized = tokenizer_exp(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(model.device)

                outputs = model_exp.generate(
                    input_ids=batch_tokenized.input_ids,
                    attention_mask=batch_tokenized.attention_mask,
                    max_length=64,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    pad_token_id=tokenizer_exp.pad_token_id,
                )

                for j, text in enumerate(batch_texts):
                    expanded_text = tokenizer_exp.decode(
                        outputs[j], skip_special_tokens=True
                    )
                    batch_texts[j] = f"{text}[SEP]{expanded_text}"

            batch_encoded = model.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)

            impacts = model.get_token_impacts(
                batch_encoded["input_ids"], batch_encoded["attention_mask"]
            )

            for j, text in enumerate(batch_texts):
                tokens = model.tokenizer.convert_ids_to_tokens(
                    batch_encoded["input_ids"][j].tolist()
                )
                token_impacts = impacts[j].cpu().numpy()

                impact_dict = {}
                for token, impact, token_id in zip(tokens, token_impacts, batch_encoded["input_ids"][j].tolist()):
                    if token not in ["[PAD]", "[CLS]", "[SEP]"] and impact > 0:
                        if token in impact_dict:
                            continue
                        else:
                            impact_dict[token_id] = float(impact)

                if quantize_bits > 0:
                    impact_tensor = torch.tensor(list(impact_dict.values()))
                    quantized_tensor = quantize_scores(
                        impact_tensor, bits=quantize_bits
                    )
                    for k, token_id in enumerate(impact_dict.keys()):
                        impact_dict[token_id] = int(quantized_tensor[k].item())

                all_results[ids[i + j]] = impact_dict

        if doc_expansion:
            del model_exp
            del tokenizer_exp
            torch.cuda.empty_cache()

    return all_results
