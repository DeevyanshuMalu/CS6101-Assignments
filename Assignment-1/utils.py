import random
import numpy as np

random.seed(42)
np.random.seed(42)

def get_candidate_ids(searcher, query):
	candidates_ids = [hit.docid for hit in searcher.search(query, k=200)]
	return candidates_ids

def get_val_eval_ids(searcher, query):
    val_eval_ids = [hit.docid for hit in searcher.search(query, k=2000)]
    random.shuffle(val_eval_ids)
    return val_eval_ids[:1000], val_eval_ids[1000:2000]

def get_relevance_score(index_reader, doc_id, query):
	return index_reader.compute_query_document_score(doc_id, query)

def get_bm25_vector(index_reader, doc_id):
	tf = index_reader.get_document_vector(doc_id)
	return {term: index_reader.compute_bm25_term_weight(doc_id, term, analyzer=None) for term in tf.keys()}

def get_similarity_score(all_bm25_vecs, doc_id_i, doc_id_j):
	vec_i = all_bm25_vecs[doc_id_i]
	vec_j = all_bm25_vecs[doc_id_j]
	common_terms = set(vec_i.keys()).intersection(set(vec_j.keys()))
	if not common_terms:
		return 0.0
	vec_i_l2_norm = np.sqrt(sum(v**2 for v in vec_i.values()))
	vec_j_l2_norm = np.sqrt(sum(v**2 for v in vec_j.values()))
	sim = sum(vec_i[term] * vec_j[term] for term in common_terms)
	sim /= (vec_i_l2_norm * vec_j_l2_norm)
	return sim

def get_c(i, S, c_is, similarity_matrix, fast=False):
	new_doc_id = S[-1]
	if fast:
		c_i = max(c_is[i], similarity_matrix[(min(i, new_doc_id), max(i, new_doc_id))])
	else:
		c_i = max([similarity_matrix[(min(i, j), max(i, j))] for j in S] or [0.0])
	return c_i