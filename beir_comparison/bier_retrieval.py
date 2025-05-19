from typing import Dict, List

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import logging

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search import BaseSearch
from tqdm import tqdm

from BM25_PIR.dataloader import process_text

from bm25s import BM25
import bm25s

class RegularBM25(BaseSearch):
    def search(self, corpus: dict[str, dict[str, str]],
               queries: dict[str, str],
               top_k: int, score_function, **kwargs) -> dict[str, dict[str, float]]:

        bm25 = BM25(backend=self.backend)
        corpus_token = process_text(corpus, self.stemmer)
        bm25.index(corpus_token)

        doc_ids = list(corpus.keys())  # index → doc-ID
        results = {}

        for qid, query in queries.items():
            query_tok = bm25s.tokenize(query, self.stemmer)
            idx_lists, score_lists = bm25.retrieve(query_tok, k=top_k)

            ranked = {}
            for rank, doc_idx in enumerate(idx_lists[0]):
                doc_id = doc_ids[doc_idx]  # map back
                ranked[str(doc_id)] = float(score_lists[0][rank])  # cast to float

            results[str(qid)] = ranked  # outer key must be str, too

        return results

    def __init__(self, stemmer, backend: str = "numba", k: int = 10):
        super().__init__()
        self.stemmer = stemmer
        self.backend = backend
        self.k = k
        logging.info(f"Initialized TokenizedBM25Retriever with backend={backend} and k={k}")

    def build_index(self, corpus, stemmer):
        tokens = []
        doc_ids = []
        for doc_id, doc in corpus.items():  # one single pass
            doc_ids.append(doc_id)
            tokens.append(bm25s.tokenize(doc["text"], stemmer))
        return tokens, doc_ids


class TokenizedBM25Retriever(BaseSearch):
    def search(self, corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int, score_function, **kwargs) -> dict[
        str, dict[str, float]]:
        results = {}

        bm25_search = BM25(backend=self.backend)
        corpus_token = process_text(corpus, self.stemmer)
        bm25_search.index(corpus_token)
        # Convert corpus to list form to index by int
        doc_ids = list(corpus.keys())
        texts = [corpus[doc_id]["text"] for doc_id in doc_ids]

        for qid, query in queries.items():
            print(f"\n\nQuery: {qid} out of {1390}\n")
            new_corpus = []

            # Word-level tokenized retrieval
            for word in query.split():
                question_tokens = bm25s.tokenize(word, self.stemmer)
                if not question_tokens.vocab:
                    continue
                search_results, _ = bm25_search.retrieve(question_tokens, k=top_k)
                for result in search_results[0]:
                    new_corpus.append(texts[result])

            # Re-index the merged results
            new_retriever = BM25(backend=self.backend)
            corpus_token = process_text(new_corpus, self.stemmer)
            new_retriever.index(corpus_token)

            # Final query on the merged sub-corpus
            question_tokens = bm25s.tokenize(query, self.stemmer)
            search_results, scores = new_retriever.retrieve(question_tokens, k=top_k)

            # Rank mapping back to original corpus
            ranked_docs = {}
            for idx, doc_idx in enumerate(search_results[0]):
                doc_text = new_corpus[doc_idx]
                try:
                    original_id = next(key for key, value in corpus.items() if value["text"] == doc_text)
                    ranked_docs[original_id] = float(scores[0][idx])
                except StopIteration:
                    continue

            results[str(qid)] = ranked_docs

        return results

    def __init__(self, stemmer, backend: str = "numba", k: int = 10):
        super().__init__()
        self.stemmer = stemmer
        self.backend = backend
        self.k = k
        logging.info(f"Initialized TokenizedBM25Retriever with backend={backend} and k={k}")


if __name__ == "__main__":
    #### Load dataset
    dataset = "msmarco"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = f"/mnt/nextcloud/10TB-STHDD/datasets/{dataset}"
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    model = TokenizedBM25Retriever(stemmer="english")
    model = RegularBM25(stemmer="english")

    retriever = EvaluateRetrieval(model, k_values=[1,5,10,100])

    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    print(f"Retriever evaluation for k in: {retriever.k_values}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    print(f"NDCG@{retriever.k_values} : {ndcg}")
    print(f"MAP@{retriever.k_values} : {_map}")
    print(f"Recall@{retriever.k_values} : {recall}")
    print(f"Precision@{retriever.k_values} : {precision}")
