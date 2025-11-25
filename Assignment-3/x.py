from model import DeepImpactModel
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dummy_model = DeepImpactModel(
#     device=device,
# )

# total_params = sum(p.numel() for p in dummy_model.parameters())
# total_params_mlp = sum(
#     p.numel() for p in dummy_model.impact_encoder.parameters()
# )
# print(f"Total parameters: {total_params}")
# print(f"Total parameters in MLP impact encoder: {total_params_mlp}")

tokenizer_exp = T5Tokenizer.from_pretrained(
    "castorini/doc2query-t5-base-msmarco"
)
model_exp = T5ForConditionalGeneration.from_pretrained(
    "castorini/doc2query-t5-base-msmarco"
).to(device)
model_exp.eval()

batch_docs = [
    "Which one dissolve in water quikly sugar, salt, methane and carbon di oxide?",
    "Method to find separation of slits using fresnel biprism?",
    "What career advice would you give to someone who wants to be a financial analyst?",
]

batch_tokenized = tokenizer_exp(
    batch_docs,
    padding=True,
    truncation=True,
    max_length=160,
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

expanded_texts = [tokenizer_exp.decode(
    outputs[i], skip_special_tokens=True
) for i in range(outputs.size(0))]

print("Original documents:", batch_docs)
print("Expanded text:", expanded_texts)