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
    # queries = queries / (torch.norm(queries, dim=1, keepdim=True) + 1e-8)
    # docs = docs / (torch.norm(docs, dim=1, keepdim=True) + 1e-8)

    transformed_queries = torch.matmul(queries, A.t())  # (batch_size, embed_dim)
    transformed_docs = torch.matmul(A, docs.t())  # (embed_dim, batch_size)
    scores = lambda_ * torch.sum(
        transformed_queries * transformed_docs.t(), dim=1
    )  # (batch_size,)
    scores += torch.sum(queries * docs, dim=1)  # (batch_size,)
    return scores


def all_sim(queries, docs, A, lambda_):
    """Compute similarity scores between queries and documents using translation matrix A."""
    # queries: (batch_size, vocab_size)
    # docs: (batch_size, vocab_size)
    # A: (embed_dim, vocab_size)
    # queries = queries / (torch.norm(queries, dim=1, keepdim=True) + 1e-8)
    # docs = docs / (torch.norm(docs, dim=1, keepdim=True) + 1e-8)

    transformed_queries = torch.matmul(queries, A.t())  # (batch_size, embed_dim)
    transformed_docs = torch.matmul(A, docs.t())  # (embed_dim, batch_size)
    scores = lambda_ * torch.matmul(
        transformed_queries, transformed_docs
    )  # (batch_size, batch_size)
    scores += torch.matmul(queries, docs.t())  # (batch_size, batch_size)
    return scores


def sim_sparse(queries, docs, A, lambda_, sparse_indices_tensor):
    """Compute similarity scores between queries and documents using sparse translation matrix.

    Args:
        queries: (batch_size, vocab_size)
        docs: (batch_size, vocab_size)
        A: (embed_dim, vocab_size)
        lambda_: scaling factor
        sparse_indices_tensor: a tensor of shape (num_sparse, 2) indicating non-zero positions in T matrix
    """
    batch_size, vocab_size = queries.shape

    if sparse_indices_tensor is None or sparse_indices_tensor.numel() == 0:
        return torch.zeros(batch_size, device=queries.device)

    # Use sparse_indices_tensor directly for vectorized operations
    i_indices = sparse_indices_tensor[:, 0]  # (num_sparse,)
    j_indices = sparse_indices_tensor[:, 1]  # (num_sparse,)

    # Gather relevant query and doc values
    query_vals = queries[:, i_indices]  # (batch_size, num_sparse)
    doc_vals = docs[:, j_indices]  # (batch_size, num_sparse)

    # Identity component: sum over sparse positions
    identity_scores = torch.sum(query_vals * doc_vals, dim=1)  # (batch_size,)

    # A^T A component: compute (A^T A)[i,j] for all sparse positions
    A_i = A[:, i_indices]  # (embed_dim, num_sparse)
    A_j = A[:, j_indices]  # (embed_dim, num_sparse)
    ata_values = torch.sum(A_i * A_j, dim=0)  # (num_sparse,)

    # Translation component contribution
    translation_contrib = lambda_ * torch.sum(
        query_vals * doc_vals * ata_values.unsqueeze(0), dim=1
    )

    return identity_scores + translation_contrib


def all_sim_sparse_old(queries, docs, A, lambda_, sparse_indices_tensor):
    """Compute all pairwise similarity scores using sparse translation matrix.

    Args:
        queries: (batch_size, vocab_size)
        docs: (batch_size, vocab_size)
        A: (embed_dim, vocab_size)
        lambda_: scaling factor
        sparse_indices_tensor: a tensor of shape (num_sparse, 2) indicating non-zero positions in T matrix
    """
    query_batch_size, vocab_size = queries.shape
    doc_batch_size = docs.shape[0]

    if sparse_indices_tensor is None or sparse_indices_tensor.numel() == 0:
        return torch.zeros(query_batch_size, doc_batch_size, device=queries.device)

    # Initialize result tensor
    scores = torch.zeros(query_batch_size, doc_batch_size, device=queries.device)

    # Process sparse indices in chunks to avoid memory explosion
    chunk_size = min(
        1000, sparse_indices_tensor.shape[0]
    )  # Adjust chunk size based on memory

    for chunk_start in range(0, sparse_indices_tensor.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, sparse_indices_tensor.shape[0])
        chunk_indices = sparse_indices_tensor[chunk_start:chunk_end]  # (chunk_size, 2)

        i_indices = chunk_indices[:, 0]  # (chunk_size,)
        j_indices = chunk_indices[:, 1]  # (chunk_size,)

        # Gather relevant query and doc values for this chunk
        query_vals = queries[:, i_indices]  # (batch_size, chunk_size)
        doc_vals = docs[:, j_indices]  # (batch_size, chunk_size)

        # Identity component for this chunk
        # query_vals: (batch_size, chunk_size) -> unsqueeze(1) -> (batch_size, 1, chunk_size)
        # doc_vals: (batch_size, chunk_size) -> unsqueeze(0) -> (1, batch_size, chunk_size)
        identity_contrib = torch.sum(
            query_vals.unsqueeze(1)
            * doc_vals.unsqueeze(
                0
            ),  # (batch_size, 1, chunk_size) * (1, batch_size, chunk_size)
            dim=2,
        )  # (batch_size, batch_size)

        # A^T A component for this chunk
        A_i = A[:, i_indices]  # (embed_dim, chunk_size)
        A_j = A[:, j_indices]  # (embed_dim, chunk_size)
        ata_values = torch.sum(A_i * A_j, dim=0)  # (chunk_size,)

        # Translation component for this chunk
        # ata_values: (chunk_size,) -> unsqueeze(0).unsqueeze(0) -> (1, 1, chunk_size)
        print(
            query_vals.unsqueeze(1).shape,
            doc_vals.unsqueeze(0).shape,
            ata_values.unsqueeze(0).unsqueeze(0).shape,
        )
        translation_contrib = lambda_ * torch.sum(
            query_vals.unsqueeze(1)
            * doc_vals.unsqueeze(0)
            * ata_values.unsqueeze(0).unsqueeze(0),
            dim=2,
        )  # (batch_size, batch_size)
        # Accumulate results
        scores += identity_contrib + translation_contrib

    return scores


def all_sim_sparse(
    queries, docs, A, lambda_, sparse_indices_tensor, chunk_size=2000, sub_batch=128
):
    """
    Memory-efficient all-pairs similarity using sparse translation matrix.
    """
    device = queries.device
    Q, V = queries.shape
    D = docs.shape[0]

    if sparse_indices_tensor is None or sparse_indices_tensor.numel() == 0:
        return torch.zeros(Q, D, device=device)

    scores = torch.zeros(Q, D, device=device)

    # Process sparse indices in manageable chunks
    for chunk_start in range(0, sparse_indices_tensor.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, sparse_indices_tensor.shape[0])
        chunk_indices = sparse_indices_tensor[chunk_start:chunk_end]
        i_idx, j_idx = chunk_indices[:, 0], chunk_indices[:, 1]

        # Precompute A^T A values for this chunk
        A_i = A[:, i_idx]  # (E, chunk)
        A_j = A[:, j_idx]  # (E, chunk)
        ata_values = torch.sum(A_i * A_j, dim=0)  # (chunk,)

        # Extract corresponding doc values once
        doc_vals = docs[:, j_idx]  # (D, chunk)
        doc_T = doc_vals.T  # (chunk, D)

        # Precompute weighted doc values for translation component
        weighted_doc_T = doc_T * ata_values.unsqueeze(1)  # (chunk, D)

        # Process queries in sub-batches to limit memory
        for q_start in range(0, Q, sub_batch):
            q_end = min(q_start + sub_batch, Q)
            q_vals = queries[q_start:q_end, i_idx]  # (sub_Q, chunk)

            # Identity contribution: sum over sparse positions
            identity_contrib = q_vals @ doc_T  # (sub_Q, D)

            # Translation contribution: weighted sum
            translation_contrib = lambda_ * (q_vals @ weighted_doc_T)  # (sub_Q, D)

            scores[q_start:q_end] += identity_contrib + translation_contrib

    return scores


def triplet_loss(
    queries, good_docs, bad_docs, A, lambda_, margin=1.0, sparse_indices_tensor=None
):
    if sparse_indices_tensor is not None:
        L = (
            margin
            + sim_sparse(queries, bad_docs, A, lambda_, sparse_indices_tensor)
            - sim_sparse(queries, good_docs, A, lambda_, sparse_indices_tensor)
        )
    else:
        L = (
            margin
            + sim(queries, bad_docs, A, lambda_)
            - sim(queries, good_docs, A, lambda_)
        )
    return torch.mean(torch.clamp(L, min=0))


def infoNCE_loss(queries, good_docs, A, lambda_, tau=1.0, sparse_indices_tensor=None):
    if sparse_indices_tensor is not None:
        all_sims = all_sim_sparse(
            queries, good_docs, A, lambda_, sparse_indices_tensor
        )  # (batch_size, batch_size)
    else:
        all_sims = all_sim(queries, good_docs, A, lambda_)  # (batch_size, batch_size)
    all_sims = all_sims / tau
    log_probs = torch.log_softmax(all_sims, dim=1)  # (batch_size, batch_size)
    return -torch.mean(torch.diagonal(log_probs))  # Average


def DCG_at_k(good_doc_ids, ranked_doc_ids):
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
