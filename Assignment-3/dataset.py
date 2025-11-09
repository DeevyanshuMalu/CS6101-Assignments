from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch


class TriplesDataset(Dataset):
    def __init__(
        self,
        training_data,
        id_to_query,
        id_to_corpus,
        doc_expansion=False,
        device=None,
        max_length=160,
        expansion_batch_size=32,
    ):
        self.rows = []

        if doc_expansion:
            tokenizer_exp = T5Tokenizer.from_pretrained(
                "castorini/doc2query-t5-base-msmarco"
            )
            model_exp = T5ForConditionalGeneration.from_pretrained(
                "castorini/doc2query-t5-base-msmarco"
            ).to(device)
            model_exp.eval()

            unique_docs = set()
            for row in training_data:
                query_id, pos_id, neg_id = row
                unique_docs.add(pos_id)
                unique_docs.add(neg_id)

            unique_docs = list(unique_docs)

            expanded_docs = self.expand_documents(
                unique_docs,
                id_to_corpus,
                tokenizer_exp,
                model_exp,
                device,
                max_length,
                expansion_batch_size,
            )

            del model_exp
            del tokenizer_exp
            torch.cuda.empty_cache()

        for row in training_data:
            query_id, pos_id, neg_id = row
            query = id_to_query[query_id]

            if doc_expansion:
                pos = expanded_docs[pos_id]
                neg = expanded_docs[neg_id]
            else:
                pos = id_to_corpus[pos_id]
                neg = id_to_corpus[neg_id]

            self.rows.append((query, pos, neg))

    def expand_documents(
        self,
        doc_ids,
        id_to_corpus,
        tokenizer_exp,
        model_exp,
        device,
        max_length,
        batch_size,
    ):
        expanded_docs = {}

        with torch.no_grad():
            for i in tqdm(
                range(0, len(doc_ids), batch_size), desc="Expanding documents"
            ):
                batch_ids = doc_ids[i : i + batch_size]
                batch_docs = [id_to_corpus[doc_id] for doc_id in batch_ids]

                batch_tokenized = tokenizer_exp(
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                outputs = model_exp.generate(
                    input_ids=batch_tokenized.input_ids,
                    attention_mask=batch_tokenized.attention_mask,
                    max_length=64,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    pad_token_id=tokenizer_exp.pad_token_id,
                )

                for j, doc_id in enumerate(batch_ids):
                    original_doc = batch_docs[j]
                    expanded_text = tokenizer_exp.decode(
                        outputs[j], skip_special_tokens=True
                    )
                    expanded_docs[doc_id] = f"{original_doc}[SEP]{expanded_text}"

        return expanded_docs

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]
