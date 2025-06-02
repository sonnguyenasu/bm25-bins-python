#!/usr/bin/env python3
# evaluate_tokenized_bm25_msmarco.py

import logging
import os
import pathlib

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search import BaseSearch
from beir.retrieval.evaluation import EvaluateRetrieval

import elasticsearch
# _original_es_init = elasticsearch.Elasticsearch.__init__
# def _patched_es_init(self, *args, timeout=None, **kwargs):
#     # remap BEIR’s timeout → request_timeout
#     if timeout is not None:
#         kwargs["request_timeout"] = timeout
#     return _original_es_init(self, *args, **kwargs)
# elasticsearch.Elasticsearch.__init__ = _patched_es_init


class RegularBM25(BaseSearch):
    def __init__(self,
                 hostname: str = "localhost:9200",
                 index_name: str = "msmarco",
                 initialize: bool = True,
                 shards: int = 1):
        super().__init__()
        self.bm25 = BM25Search(
            index_name=index_name,
            hostname=hostname,
            initialize=initialize,
            number_of_shards=shards,
        )
        logging.info(f"[RegularBM25] ES host={hostname}, index={index_name}")

    def search(self,
               corpus: dict[str, dict[str, str]],
               queries: dict[str, str],
               top_k: int,
               score_function,
               **kwargs
               ) -> dict[str, dict[str, float]]:
        # delegate to BEIR’s BM25Search on the whole corpus
        return self.bm25.search(corpus, queries, top_k, score_function)


# -----------------------------------------------------------------------------
# 2) Tokenized BM25: per-word hits → merge → rerank
# -----------------------------------------------------------------------------
class TokenizedBM25Retriever(BaseSearch):
    def __init__(self,
                 hostname: str = "localhost:9200",
                 index_name: str = "msmarco",
                 initialize: bool = True,
                 shards: int = 1):
        super().__init__()
        self.bm25 = BM25Search(
            index_name=index_name,
            hostname=hostname,
            initialize=initialize,
            number_of_shards=shards,
        )
        logging.info(f"[TokenizedBM25] ES host={hostname}, index={index_name}")

    def search(self,
               corpus: dict[str, dict[str, str]],
               queries: dict[str, str],
               top_k: int,
               score_function,
               **kwargs
               ) -> dict[str, dict[str, float]]:

        results: dict[str, dict[str, float]] = {}

        for qid, query in queries.items():
            # 1) word-level retrieval on the full index
            merged_doc_ids: set[str] = set()
            for word in query.split():
                if not word.strip():
                    continue
                hits = self.bm25.search(corpus, {qid: word}, top_k, score_function)[qid]
                merged_doc_ids.update(hits.keys())

            # 2) build the sub-corpus of all docs seen above
            sub_corpus = {doc_id: corpus[doc_id]
                          for doc_id in merged_doc_ids}

            # 3) final full-query retrieval on that sub-corpus
            final_hits = self.bm25.search(sub_corpus, {qid: query}, top_k, score_function)[qid]
            results[qid] = final_hits

        return results


def main():

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    dataset = "msmarco"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent, "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


    model = TokenizedBM25Retriever()

    retriever = EvaluateRetrieval(model, k_values=[1, 5, 10, 100])
    results = retriever.retrieve(corpus, queries)

    logging.info(f"Evaluation for k in {retriever.k_values}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    print(f"NDCG@{retriever.k_values}    : {ndcg}")
    print(f"MAP@{retriever.k_values}     : {_map}")
    print(f"Recall@{retriever.k_values}  : {recall}")
    print(f"Precision@{retriever.k_values}: {precision}")


if __name__ == "__main__":
    main()
