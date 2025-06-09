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

import re
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)   # letters + digits

def tokenize(text: str) -> list[str]:
    """Lower-case, drop punctuation, collapse whitespace."""
    return TOKEN_RE.findall(text.lower())
    return text.split(" ")

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
                 shards: int = 1,
                 k = 10):
        super().__init__()

        self.index_name = index_name
        self.hostname = hostname
        self.initialize = initialize
        self.shards = shards
        self.k = k

        logging.info(f"[TokenizedBM25] ES host={hostname}, index={index_name}")
        self.n = n

    def search(self,
               corpus: dict[str, dict[str, str]],
               queries: dict[str, str],
               top_k: int,
               score_function,
               **kwargs
               ) -> dict[str, dict[str, float]]:

        original_queries = deepcopy(queries)
        original_queries = {qid: " ".join(tokenize(q)) for qid, q in original_queries.items()}
        original_corpus = deepcopy(corpus)

        # corpus is a str of DOC IDs mapping to a dict of 'text' and 'title'
        # queries are QUERY IDs mapping to the query

        # result is supposed to be QID mapping to DOC ID and its BM25 score

        print("Beginning ngram BM25")

        keywords = set()

        for item in corpus.values():
            keywords.update(tokenize(item["title"]))
            keywords.update(tokenize(item["text"]))

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
        cooling = 1.1
        logging.warning("Keywords: {}".format(len(keywords))) # elasticsearch uses info when it should use debug... so we ahve to use warn!
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
                    key = tokenize(words)[0]  # first token
                    ngram_lookup[key] = words
                    queries[key] = words
                    break

                words = ""
                for i in range(self.n - 1):
                    word = keywords.pop()
                    words = words + (word + " ")

                word = keywords.pop()
                words = words + word
                ngram_lookup[tokenize(words)[0]] = words
                queries[tokenize(words)[0]] = words

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
                words = tokenize(original_query)
                best_random_word = max(words, key=lambda w: unigram_scores.get(w, -1))
                max_unigram_score = unigram_scores.get(best_random_word)
                ngram_score = sum(hits[key].values())

                if ngram_score >= max_unigram_score * cooling:
                    final_hits[key] = hits[key]
                else:
                    for word in words:
                        leftovers.add(word)
            logging.warning("Leftovers: {}".format(len(leftovers)))
            keywords = leftovers
            cooling -= 0.05


        new_lookup = {}
        new_corpus = {}
        for key, hits in final_hits.items():  # ngramlookup has some dead values
            for word in tokenize(ngram_lookup[key]):
                new_lookup[word] = hits

        for qid, query_text in original_queries.items():
            tokens = tokenize(query_text)

            for token in tokens:
                results = new_lookup.get(token)

                if results is None: # this word doesn't appear in our corpus
                    logging.warning(f"missing word: {qid}, {token}")
                    continue

                for doc_id, score in results.items():
                    doc_items = original_corpus.get(doc_id)
                    new_corpus[doc_id] = doc_items

        final_bm25 = BM25Search(
            index_name=self.index_name,
            hostname=self.hostname,
            initialize=self.initialize,
            number_of_shards=self.shards,
            retry_on_timeout=True,
            timeout=600
        )
        results = final_bm25.search(new_corpus, original_queries, top_k, score_function)
        return results


class ngramBM25Retriever_freq(BaseSearch):
    def __init__(self,
                 hostname: str = "http://fedora-ripley.tail0c8c1f.ts.net:9200",
                 index_name: str = "msmarco",
                 initialize: bool = True,
                 n: int = 2,
                 shards: int = 1,
                 k = 10,
                 frequency = 10,):
        super().__init__()

        self.index_name = index_name
        self.hostname = hostname
        self.initialize = initialize
        self.shards = shards
        self.k = k

        logging.info(f"[TokenizedBM25] ES host={hostname}, index={index_name}")
        self.n = n
        self.frequency = frequency

    def search(self,
               corpus: dict[str, dict[str, str]],
               queries: dict[str, str],
               top_k: int,
               score_function,
               **kwargs
               ) -> dict[str, dict[str, float]]:

        original_queries = deepcopy(queries)
        original_queries = {qid: " ".join(tokenize(q)) for qid, q in original_queries.items()}
        original_corpus = deepcopy(corpus)

        # corpus is a str of DOC IDs mapping to a dict of 'text' and 'title'
        # queries are QUERY IDs mapping to the query

        # result is supposed to be QID mapping to DOC ID and its BM25 score

        print("Beginning ngram BM25")

        keywords = set()

        for item in corpus.values():
            keywords.update(tokenize(item["title"]))
            keywords.update(tokenize(item["text"]))

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

        # We hope that this will produce at least 1 doc per keyword
        unigram_hits = unigram_bm25.search(corpus, queries, len(corpus.keys()), score_function)


        unigram_scores = {}
        # A dictionary mapping docID to keywords that appear and their score
        docs_with_scores: dict[str, dict[str, float]] = {}


        for key, val in unigram_hits.items():

            for doc_id in val.keys():
                if doc_id not in docs_with_scores:
                    docs_with_scores[doc_id] = {}
                docs_with_scores[doc_id][key] = val[doc_id]

            score = sum(val.values())
            logging.debug(f"{key}, {score}")
            unigram_scores[key] = score


        # Mapping from doc id to the list of self.frequency number of ngrams
        requested_words: dict[str, list[str]] = {}

        for doc_id, score in docs_with_scores.items():
            top_keywords = sorted(score.items(), key=lambda x: x[1], reverse=True)[:self.frequency]
        





        new_lookup = {}
        new_corpus = {}
        for key, hits in final_hits.items():  # ngramlookup has some dead values
            for word in tokenize(ngram_lookup[key]):
                new_lookup[word] = hits

        for qid, query_text in original_queries.items():
            tokens = tokenize(query_text)

            for token in tokens:
                results = new_lookup.get(token)

                if results is None: # this word doesn't appear in our corpus
                    logging.warning(f"missing word: {qid}, {token}")
                    continue

                for doc_id, score in results.items():
                    doc_items = original_corpus.get(doc_id)
                    new_corpus[doc_id] = doc_items

        final_bm25 = BM25Search(
            index_name=self.index_name,
            hostname=self.hostname,
            initialize=self.initialize,
            number_of_shards=self.shards,
            retry_on_timeout=True,
            timeout=600
        )
        results = final_bm25.search(new_corpus, original_queries, top_k, score_function)
        return results


def main():

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.WARN,
        handlers=[LoggingHandler()],
    )

    dataset = "quora"
    dataset = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    # out_dir = os.path.join(pathlib.Path(__file__).parent, "datasets")
    out_dir = "/home/yelnat/Documents/Nextcloud/10TB-STHDD/Sync-Folder-STHDD/datasets"
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    for i in [1, 2, 3, 4, 5, 10, 20]:

        print("======================= RESULTS FOR n = {i} =======================".format(i=i))

        model = ngramBM25Retriever_freq(n=i)
        # model = RegularBM25()

        retriever = EvaluateRetrieval(model, k_values=[1000])
        results = retriever.retrieve(corpus, queries)

        logging.info(f"Evaluation for k in {retriever.k_values}")
        ndcg, map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        mrr = retriever.evaluate_custom(qrels, results, k_values=retriever.k_values, metric="mrr")

        print(f"NDCG@{retriever.k_values}    : {ndcg}")
        print(f"MAP@{retriever.k_values}     : {map}")
        print(f"Recall@{retriever.k_values}  : {recall}")
        print(f"Precision@{retriever.k_values}: {precision}")
        print(f"MRR@{retriever.k_values}     : {mrr}")

if __name__ == "__main__":
    main()
