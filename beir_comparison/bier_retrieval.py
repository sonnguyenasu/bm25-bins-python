#!/usr/bin/env python3
# evaluate_tokenized_bm25_msmarco.py

import logging
import os
import pathlib
import random
from collections import defaultdict
from copy import deepcopy
from itertools import combinations

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search import BaseSearch
from beir.retrieval.evaluation import EvaluateRetrieval


class RegularBM25(BaseSearch):
    def __init__(self,
                 hostname: str = "http://fedora-ripley.tail0c8c1f.ts.net:9200",
                 index_name: str = "msmarco",
                 initialize: bool = True,
                 shards: int = 1):
        super().__init__()
        self.bm25 = BM25Search(
            index_name=index_name,
            hostname=hostname,
            initialize=initialize,
            number_of_shards=shards,
            timeout=600,
            retry_on_timeout=True
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
        results = self.bm25.search(corpus, queries, top_k, score_function)
        return results



class TokenizedBM25Retriever(BaseSearch):
    def __init__(self,
                 hostname: str = "http://fedora-ripley.tail0c8c1f.ts.net:9200",
                 index_name: str = "msmarco",
                 initialize: bool = True,
                 shards: int = 1):
        super().__init__()
        self.bm25 = BM25Search(
            index_name=index_name,
            hostname=hostname,
            initialize=initialize,
            number_of_shards=shards,
            retry_on_timeout=True,
            timeout=600
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

        print("Beginning Tokenised BM25")

        for qid, query in queries.items():
            # 1) word-level retrieval on the full index
            merged_doc_ids: set[str] = set()
            for word in query.split():
                if not isinstance(word, str) and not word.strip():
                    continue
                hits = self.bm25.search(corpus, {qid: word}, top_k, score_function)[qid]
                merged_doc_ids.update(hits.keys())



            # 2) build the sub-corpus of all docs seen above
            sub_corpus = {doc_id: corpus[doc_id]
                          for doc_id in merged_doc_ids}

            print("Sub-corpus built")

            # 3) final full-query retrieval on that sub-corpus
            final_hits = self.bm25.search(sub_corpus, {qid: query}, top_k, score_function)[qid]
            results[qid] = final_hits

        return results


class ngramBM25Retriever(BaseSearch):
    def __init__(self,
                 hostname: str = "http://fedora-ripley.tail0c8c1f.ts.net:9200",
                 index_name: str = "msmarco",
                 initialize: bool = True,
                 n: int = 2,
                 shards: int = 1):
        super().__init__()

        self.index_name = index_name
        self.hostname = hostname
        self.initialize = initialize
        self.shards = shards

        logging.info(f"[TokenizedBM25] ES host={hostname}, index={index_name}")
        self.n = n

    def search(self,
               corpus: dict[str, dict[str, str]],
               queries: dict[str, str],
               top_k: int,
               score_function,
               **kwargs
               ) -> dict[str, dict[str, float]]:

        original_queries = queries.copy()

        # corpus is a str of DOC IDs mapping to a dict of 'text' and 'title'
        # queries are QUERY IDs mapping to the query

        # result is supposed to be QID mapping to DOC ID and its BM25 score

        print("Beginning ngram BM25")

        keywords = set()

        for item in corpus.values():
            keywords.update(item["title"].split(" "))
            keywords.update(item["text"].split(" "))

        # to ensure that no score drags any others down, we do an unigram analysis

        queries = {}
        for word in keywords:
            queries[f"{word}"] = f"{word}"

        unigram_bm25 = BM25Search(
            index_name=self.index_name,
            hostname=self.hostname,
            initialize=self.initialize,
            number_of_shards=self.shards,
            retry_on_timeout=True,
            timeout=600
        )

        unigram_hits = unigram_bm25.search(corpus, queries, top_k, score_function)

        unigram_scores = {}
        for key, val in unigram_hits.items():
            score = sum(val.values())
            logging.debug(f"{key}, {score}")
            unigram_scores[key] = score

        final_hits = {}
        ngram_lookup = {}
        while keywords:
            leftovers = set()

            queries = {}
            # Todo: The way this should work is that you just randomly choose all the words and their pairs. Then compare
            # the score to their unigram scoreas and break up any items that don't match the max unigram score
            while True:

                if len(keywords) <= self.n:
                    if len(keywords) == 0:
                        break
                    words = " ".join(keywords)
                    key = words.split(" ")[0]  # first token
                    ngram_lookup[key] = words
                    queries[key] = words
                    break

                words = ""
                for i in range(self.n - 1):
                    word = keywords.pop()
                    words = words + (word + " ")

                word = keywords.pop()
                words = words + word
                ngram_lookup[words.split(" ")[0]] = words
                queries[words.split(" ")[0]] = words

            ngram_bm25 = BM25Search(
                index_name=self.index_name,
                hostname=self.hostname,
                initialize=self.initialize,
                number_of_shards=self.shards,
                retry_on_timeout=True,
                timeout=600
            )

            hits = ngram_bm25.search(corpus, queries, top_k, score_function)

            for key in hits.keys():
                # check if each hit surpasses its unigram score
                original_query = ngram_lookup[key]

                # Find the word in the key with the highest unigram score
                words = original_query.split(" ")
                best_random_word = max(words, key=lambda w: unigram_scores.get(w, -1))
                max_unigram_score = unigram_scores.get(best_random_word)
                ngram_score = sum(hits[key].values())

                if ngram_score >= max_unigram_score:
                    final_hits[key] = hits[key]
                else:
                    # Add these back into the pool
                    leftovers.add(words)
            keywords = leftovers


        aggregated_results: dict[str, dict[str, float]] = {}

        for qid, query_text in original_queries.items():
            tokens = query_text.split()  # whitespace tokenisation
            doc_scores: dict[str, float] = {}

            # slide a window of size self.n over the query
            for i in range(len(tokens) - self.n + 1):
                ngram = " ".join(tokens[i: i + self.n])
                key = ngram.split(" ")[0]  # ← same transform you used earlier

                # if we produced hits for that key, merge them in
                if key in final_hits:
                    for doc_id, score in final_hits[key].items():
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score  # sum; use max() if preferred

            # optional fallback: if the query had no matching n-grams, fall back to its best unigram
            if not doc_scores:
                for token in tokens:
                    if token in unigram_hits:
                        for doc_id, score in unigram_hits[token].items():
                            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                        break  # take only the first token that exists

            # keep only the top-k documents for this query
            if doc_scores:
                top_docs = dict(
                    sorted(doc_scores.items(), key=lambda t: t[1], reverse=True)[: top_k]
                )
                aggregated_results[qid] = top_docs

        return aggregated_results




def main():

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[LoggingHandler()],
    )

    dataset = "trec-covid"
    dataset = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent, "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


    model = ngramBM25Retriever(n=5)
    #model = RegularBM25()

    retriever = EvaluateRetrieval(model, k_values=[10])
    results = retriever.retrieve(corpus, queries)

    logging.info(f"Evaluation for k in {retriever.k_values}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    print(f"NDCG@{retriever.k_values}    : {ndcg}")
    print(f"MAP@{retriever.k_values}     : {_map}")
    print(f"Recall@{retriever.k_values}  : {recall}")
    print(f"Precision@{retriever.k_values}: {precision}")


if __name__ == "__main__":
    main()
