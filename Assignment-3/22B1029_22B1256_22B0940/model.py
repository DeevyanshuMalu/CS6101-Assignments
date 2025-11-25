import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class DeepImpactModel(nn.Module):
    def __init__(self, device, mlp_hidden=256, freeze_bert=True):
        super().__init__()
        self.device = device
        self.bert = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.embed_dim = self.bert.config.hidden_size
        self.impact_encoder = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
            nn.ReLU(),
        ).to(device)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def get_token_impacts(self, input_ids, attention_mask):
        with torch.set_grad_enabled(self.training):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            sequence_output = outputs.hidden_states[
                -1
            ]  # (batch_size, seq_len, hidden_size)

            impacts = self.impact_encoder(sequence_output)  # (batch_size, seq_len, 1)
            impacts = impacts.squeeze(-1)  # (batch_size, seq_len)

            impacts = impacts * attention_mask.float()

        return impacts

    def get_query_doc_scores(self, query_batch, doc_batch, doc_impacts):
        batch_size = query_batch["input_ids"].size(0)
        scores = []

        for i in range(batch_size):
            q_tokens = self.tokenizer.convert_ids_to_tokens(
                query_batch["input_ids"][i].tolist()
            )
            doc_tokens = self.tokenizer.convert_ids_to_tokens(
                doc_batch["input_ids"][i].tolist()
            )
            doc_impact_scores = doc_impacts[i]

            score = self.calculate_single_score(q_tokens, doc_tokens, doc_impact_scores)
            scores.append(score)

        return torch.stack(scores)  # (batch_size,)

    def calculate_single_score(self, q_tokens, doc_tokens, doc_impacts):
        doc_token_impacts = {}
        for j, (token, impact) in enumerate(zip(doc_tokens, doc_impacts)):
            if token not in ["[PAD]", "[CLS]", "[SEP]"]:
                if token in doc_token_impacts:
                    continue
                else:
                    doc_token_impacts[token] = impact

        score = torch.tensor(0.0, device=self.device)
        for token in q_tokens:
            if token in doc_token_impacts and token not in ["[PAD]", "[CLS]", "[SEP]"]:
                score += doc_token_impacts[token]

        return score
