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
        self.bm25 = BM25Search(
            index_name=index_name,
            hostname=hostname,
            initialize=initialize,
            number_of_shards=shards,
            retry_on_timeout=True,
            timeout=600
        )
        logging.info(f"[TokenizedBM25] ES host={hostname}, index={index_name}")
        self.n = n

    def search(self,
               corpus: dict[str, dict[str, str]],
               queries: dict[str, str],
               top_k: int,
               score_function,
               **kwargs
               ) -> dict[str, dict[str, float]]:

        results: dict[str, dict[str, float]] = {}

        # corpus is a str of DOC IDs mapping to a dict of 'text' and 'title'
        # queries are QUERY IDs mapping to the query

        # result is supposed to be QID mapping to DOC ID and its BM25 score

        print("Beginning ngram BM25")

        keywords = set()

        for item in corpus.values():
            keywords.update(item["title"].split())
            keywords.update(item["text"].split())

        # to ensure that no score drags any others down, we do an unigram analysis

        queries = {}
        for word in keywords:
            queries[f"{word}"] = f"{word}"

        hits = self.bm25.search(corpus, queries, top_k, score_function)

        unigram_scores = {}
        for key, val in hits.items():
            score = sum(val.values())
            logging.debug(f"{key}, {score}")
            unigram_scores[key] = score


        while keywords:
            word = keywords.pop()
            logging.debug(f"Removed:{word}")

            queries = {}
            i = 0
            for word_2 in keywords:
                #PLAIN-2: {'MED-10': 25.226639, 'MED-1193': 19.714224, 'MED-1207': 14.95818, 'MED-14': 24.935198, 'MED-2427': 18.497639, 'MED-2428': 21.614073, 'MED-2429': 25.438505, 'MED-2431': 20.217152, 'MED-3204': 18.000595, 'MED-4827': 18.001968, 'MED-4830': 15.799666}

                queries[f"{word} {word_2}"] = f"{word} {word_2}"

            hits = self.bm25.search(corpus, queries, top_k, score_function)

            # best_key = max(hits, key=lambda k: sum(hits[k].values()))
            # best_score = sum(hits[best_key].values())
            #
            # best_word = max(best_key.split(), key=lambda w: unigram_scores[w])
            # max_score = unigram_scores[best_word]
            #
            # found = False

            hits_clone = deepcopy(hits)

            while hits:
                # Find the key with the highest total hit score
                best_key = max(hits, key=lambda k: sum(hits[k].values()))
                best_score = sum(hits[best_key].values())

                # Find the word in the key with the highest unigram score
                words = best_key.split()
                best_word = max(words, key=lambda w: unigram_scores.get(w, float('-inf')))
                max_score = unigram_scores.get(best_word)

                # Remove the current best_key
                del hits[best_key]

                # Try to find a new key whose total score beats the unigram max_score
                for key, val in hits.items():
                    if sum(val.values()) > max_score:
                        # New best key found
                        best_key = key
                        break

                # No better key found, pick a random one from remaining
                if hits and best_score < max_score:
                    random_key = random.choice(list(hits.keys()))
                    random_word = max(random_key.split(), key=lambda w: unigram_scores.get(w, float('-inf')))
                    best_key = random_key


            logging.debug(f"Best key:, {best_key}")
            logging.debug(f"Total score:, {best_score}")

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




def main():

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.ERROR,
        handlers=[LoggingHandler()],
    )

    dataset = "trec-covid"
    dataset = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent, "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


    model = ngramBM25Retriever(n=2)
    # model = RegularBM25()

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
