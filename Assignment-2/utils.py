import torch
import hashlib

def text_to_id(text):
    """Return a deterministic ID for the given text."""
    # Normalize whitespace etc. to avoid accidental differences
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def sim(queries, docs, A, lambda_):
    """Compute similarity scores between queries and documents using translation matrix A."""
    # queries: (batch_size, vocab_size)
    # docs: (batch_size, vocab_size)
    # A: (embed_dim, vocab_size)
    transformed_queries = torch.matmul(queries, A.t())  # (batch_size, embed_dim)
    transformed_docs = torch.matmul(A, docs.t())  # (embed_dim, batch_size)
    scores = lambda_ * torch.sum(transformed_queries * transformed_docs.t(), dim=1)  # (batch_size,)
    scores += torch.sum(queries * docs, dim=1)  # (batch_size,)
    return scores

def all_sim(queries, docs, A, lambda_):
    """Compute similarity scores between queries and documents using translation matrix A."""
    # queries: (batch_size, vocab_size)
    # docs: (batch_size, vocab_size)
    # A: (embed_dim, vocab_size)
    transformed_queries = torch.matmul(queries, A.t())  # (batch_size, embed_dim)
    transformed_docs = torch.matmul(A, docs.t())  # (embed_dim, batch_size)
    scores = lambda_ * torch.matmul(transformed_queries, transformed_docs)  # (batch_size, batch_size)
    scores += torch.matmul(queries, docs.t())  # (batch_size, batch_size)
    return scores

def triplet_loss(queries, good_docs, bad_docs, A, lambda_, margin=1.0):
    L = margin + sim(queries, bad_docs, A, lambda_) - sim(queries, good_docs, A, lambda_)
    return torch.mean(torch.clamp(L, min=0))

def infoNCE_loss(queries, good_docs, A, lambda_, tau=1.0):
    all_sims = all_sim(queries, good_docs, A, lambda_)  # (batch_size, batch_size)
    all_sims = all_sims / tau
    log_probs = torch.log_softmax(all_sims, dim=1)  # (batch_size, batch_size)
    return -torch.mean(torch.diagonal(log_probs))  # Average

def DGC_at_k(good_doc_ids, ranked_doc_ids):
    """Compute DGC@k for a query."""
    total = 0.0
    for k, id in enumerate(ranked_doc_ids):
        if id in good_doc_ids:
            total += 1.0 / torch.log2(torch.tensor(k + 2.0))  # k is 0-indexed
    return total

def MRR_at_k(good_doc_ids, ranked_doc_ids):
    """Compute MRR for a query."""
    for k, id in enumerate(ranked_doc_ids):
        if id in good_doc_ids:
            return 1.0 / (k + 1.0)  # k is 0-indexed
    return 0.0

def recall_at_k(good_doc_ids, ranked_doc_ids):
    """Compute Recall@k for a query."""
    intersect_len = len(set(good_doc_ids).intersection(set(ranked_doc_ids)))
    return intersect_len / len(good_doc_ids)