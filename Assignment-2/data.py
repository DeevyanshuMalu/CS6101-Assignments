from torch.utils.data import Dataset
from utils import *


class HotpotQADataset(Dataset):
    def __init__(
        self, data, term_to_id, id_to_term, doc_vectors_dict, query_vectors_dict
    ):
        self.data = data
        self.term_to_id = term_to_id
        self.id_to_term = id_to_term
        self.doc_vectors_dict = doc_vectors_dict
        self.query_vectors_dict = query_vectors_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        good_doc = item["good_doc"]
        bad_doc = item["bad_doc"]

        query_id = text_to_id(query)
        good_doc_id = text_to_id(good_doc)
        bad_doc_id = text_to_id(bad_doc)

        query_vector = self.query_vectors_dict.get(query_id)
        good_doc_vector = self.doc_vectors_dict.get(good_doc_id)
        bad_doc_vector = self.doc_vectors_dict.get(bad_doc_id)

        query_tensor = torch.zeros(len(self.term_to_id))
        good_doc_tensor = torch.zeros(len(self.term_to_id))
        bad_doc_tensor = torch.zeros(len(self.term_to_id))

        for term in query_vector:
            if term in self.term_to_id:
                query_tensor[self.term_to_id[term]] = query_vector[term]
        for term in good_doc_vector:
            if term in self.term_to_id:
                good_doc_tensor[self.term_to_id[term]] = good_doc_vector[term]
        for term in bad_doc_vector:
            if term in self.term_to_id:
                bad_doc_tensor[self.term_to_id[term]] = bad_doc_vector[term]

        return {
            "query": query_tensor,
            "good_doc": good_doc_tensor,
            "bad_doc": bad_doc_tensor,
        }
