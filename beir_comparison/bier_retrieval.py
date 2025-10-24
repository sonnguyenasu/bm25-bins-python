#!/usr/bin/env python3
# evaluate_tokenized_bm25_msmarco.py
import gc
import logging
import os
import pathlib
import random
from collections import defaultdict, Counter
from contextlib import contextmanager
from copy import deepcopy
from itertools import combinations
from typing import Optional
import os, sys, contextlib
import json

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search import BaseSearch
from beir.retrieval.evaluation import EvaluateRetrieval

import re

from pyserini.analysis import Analyzer, get_lucene_analyzer   # ✔ new API
from pyserini.index import lucene                             # keep this if you still
                                                              # need Lucene readers
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

# ----- build a Krovetz‑stemmed English analyzer --------------------------
analyzer = Analyzer(get_lucene_analyzer(stemmer='krovetz'))   # one line!
# -------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Fast regex pre‑split → Lucene analyzer."""
    coarse = TOKEN_RE.findall(text.lower())
    return analyzer.analyze(" ".join(coarse))

@contextmanager
def silence_stderr():
    with open(os.devnull, "w") as devnull:
        old = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old


def tokenize(text: str) -> list[str]:
    """Lower-case, drop punctuation, collapse whitespace."""
    return TOKEN_RE.findall(text.lower())
    return text.split(" ")

class RegularBM25(BaseSearch):
    def __init__(self,
                 hostname: str = "http://localhost:9200",
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

    # No idea why this takes a century to run on MS marco...
    # def search(self,
    #            corpus: dict[str, dict[str, str]],
    #            queries: dict[str, str],
    #            top_k: int,
    #            score_function=None,  # kept for API parity; ignored locally
    #            **kwargs
    #            ) -> dict[str, dict[str, float]]:
    #     """
    #     Local BM25 over the *full* corpus.
    #     Returns BEIR‑style: {qid: {doc_id: score, …}, …}
    #     """
    #
    #     # ------------------------------------------------------------------
    #     # 1.  Build an in‑memory index exactly once for the whole corpus
    #     # ------------------------------------------------------------------
    #     doc_ids = list(corpus.keys())  # integer index → doc_id lookup
    #     logging.info(f"[RegularBM25] tokeniser starting")
    #     tokenised_docs = [tokenize(doc_to_text(corpus[d])) for d in tqdm(doc_ids)]
    #     logging.info(f"[RegularBM25] search starting")
    #     bm25 = BM25Okapi(tokenised_docs)
    #     logging.info(f"[RegularBM25] search finished")
    #     # ------------------------------------------------------------------
    #     # 2.  Score each query against that single index
    #     # ------------------------------------------------------------------
    #     final_res: dict[str, dict[str, float]] = {}
    #
    #     for qid, query_text in tqdm(queries.items()):
    #         q_tokens = tokenize(query_text)
    #
    #         scores = bm25.get_scores(q_tokens)  # np.ndarray[float]
    #         if top_k is None or top_k <= 0:
    #             # keep all docs (rare in BEIR but allowed)
    #             top_idx = np.argsort(scores)[::-1]
    #         else:
    #             top_idx = np.argsort(scores)[::-1][:top_k]
    #
    #         # BEIR wants a mapping doc_id -> score (float32 ok)
    #         final_res[qid] = {doc_ids[i]: float(scores[i])
    #                           for i in top_idx if scores[i] > 0}
    #     logging.info(f"[RegularBM25] search returning")
    #     return final_res


class TokenizedBM25Retriever(BaseSearch):
    def __init__(self,
                 hostname: str = "http://localhost:9200",
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
                 hostname: str = "http://localhost:9200",
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


        texts = []
        for item in new_corpus.values():
            texts.append(item["title"])
            texts.append(item["text"])

        # Count occurrences
        counts = Counter(texts)

        duplicates = {text: count for text, count in counts.items() if count > 1}

        # Print number of duplicated values
        print(f"Number of duplicated values in new corpus: {len(duplicates)}")

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


def weighted_top_pairs(
    requested_words: dict[str, list[tuple[str, float]]] ,
    alpha: float = 0.3,            # 0 → only score, 1 → only frequency
    tolerance: float = 0.03        # keep anything within 5 % of the best
) -> list[str]:
    """
    Combine *frequency* and *total score* into one metric.

    Each pair key gets:
        norm_freq  = freq / max_freq
        norm_score = best_score / max_score          (max across its copies)
        combined   = alpha*norm_freq + (1-alpha)*norm_score

    Return every key whose combined score is within `tolerance` of the best.
    """

    flat_pairs = (
        pair for pairs in requested_words.values() for pair in pairs
    )

    freq   = Counter()
    best_s = defaultdict(float)     # best score seen for the key

    for pair in flat_pairs:
        key   = " ".join(sorted(kw for kw, _ in pair))
        score = sum(s for _, s in pair)
        freq[key]   += 1
        best_s[key] = max(best_s[key], score)

    if not freq:
        return []

    # 2.  Normalise frequency and score terms
    max_freq  = max(freq.values())
    max_score = max(best_s.values())

    norm_combined = {}
    for key in freq:
        norm_f = freq[key]   / max_freq
        norm_s = best_s[key] / max_score
        norm_combined[key] = alpha * norm_f + (1 - alpha) * norm_s

    # 3.  Keep anything “close enough” to the best combined value
    best = max(norm_combined.values())
    band = best * tolerance
    return [k for k, v in norm_combined.items() if best - v <= band]


def top_scoring_pairs(requested_words,
                      tolerance: float = 0.03):
    """
    Return every keyword pair whose score is within `tolerance` (fraction)
    of the best score observed.
    """

    flat_pairs = [pair for pairs in requested_words.values() for pair in pairs]


    pair_to_score: dict[str, float] = defaultdict(float)
    for pair in flat_pairs:
        # alphabetical --> order-insensitive key
        key = " ".join(sorted(kw for kw, _ in pair))
        score = sum(score for _, score in pair)
        pair_to_score[key] = max(pair_to_score[key], score)

    max_score = max(pair_to_score.values())
    band      = max_score * tolerance
    return [p for p, s in pair_to_score.items() if max_score - s <= band]

def most_common_pairs(requested_words):
    """
    Return **all** keyword-score pairs that share the highest
    repeat-count across the collection, provided that count ≥ 2.
    If every pair is unique, return them all.
    """
    flat_pairs = [pair for pairs in requested_words.values() for pair in pairs]

    ngram_pairs = []

    for pair in flat_pairs:
        temp_list = [pair[0][0]]
        for i in range(1, len(pair)):
            temp_list.append(pair[i][0])
        temp_list.sort()
        temp_string = " ".join(temp_list)
        ngram_pairs.append(temp_string)

    counts = Counter(ngram_pairs)

    if not counts:
        return [pair for pair, c in count.items()]
    max_freq = max(counts.values())

    return [pair for pair, c in counts.items() if c == max_freq]



class ngramBM25Retriever_freq(BaseSearch):
    def __init__(self,
                 hostname: str = "http://localhost:9200",
                 index_name: str = "msmarco",
                 initialize: bool = True,
                 n: int = 2,
                 shards: int = 1,
                 k = 10,
                 frequency = 10,
                 method = 0,):
        super().__init__()

        self.index_name = index_name
        self.hostname = hostname
        self.initialize = initialize
        self.shards = shards
        self.k = k

        logging.info(f"[TokenizedBM25] ES host={hostname}, index={index_name}")
        self.n = n
        self.frequency = frequency
        self.method = method

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
            item["title"] = " ".join(tokenize(item["title"]))
            item["text"] = " ".join(tokenize(item["text"]))
            keywords.update(item["title"].lower().split())
            keywords.update(item["text"].lower().split())

        logging.info(f"Vocab size is: {len(keywords)}")

        # to ensure that no score drags any others down, we do an unigram analysis

        queries = {}
        for word in keywords:
            queries[f"{word}"] = f"{word}"

        texts = []
        for item in corpus.values():
            texts.append(item["title"])
            texts.append(item["text"])

        # Count occurrences
        counts = Counter(texts)

        duplicates = {text: count for text, count in counts.items() if count > 1}

        # Print number of duplicated values
        logging.info(f"Number of duplicated values: {len(duplicates)}")

        del duplicates
        del texts


        unigram_bm25 = BM25Search(
            index_name=self.index_name,
            hostname=self.hostname,
            initialize=self.initialize,
            number_of_shards=self.shards,
            retry_on_timeout=True,
            timeout=600
        )


        # We hope that this will produce at least 1 doc per keyword
        unigram_hits = unigram_bm25.search(corpus, queries, 300, score_function)


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
        requested_words: dict[str, list[tuple[str, float]]] = {}

        del unigram_hits
        gc.collect()

        # Matches the doc ID to the keywords that it requested, so we can easily remove them later
        reverse_doc_id_mathcing = {}
        new_ngrams = {}
        ngram_lookup = {}

        total_kv_pairs = sum(len(scores) for scores in docs_with_scores.values())
        pbar = tqdm(total=total_kv_pairs, desc="keywords removed")

        while True:
            if self.n == 1:
                break

            for doc_id, score in docs_with_scores.items():
                top_keywords = sorted(score.items(), key=lambda x: x[1], reverse=True)[:self.frequency]
                # test = sorted(score.items(), key=lambda x: x[1], reverse=True)
                # test_doc = corpus[doc_id]

                for keyword, score in top_keywords:
                    if keyword in reverse_doc_id_mathcing.keys():
                        reverse_doc_id_mathcing[keyword].append(doc_id)
                    else:
                        reverse_doc_id_mathcing[keyword] = [doc_id]

                requested_words[doc_id] = []
                for i in range(0, len(top_keywords)//self.n):
                    requested_words[doc_id].append(top_keywords[i * self.n:(i * self.n) + self.n])
            if not any(requested_words.values()):
                logging.info("We have no requested words")
                break
            if self.method == 0:
                mcps = most_common_pairs(requested_words)
            elif self.method == 1:
                mcps = top_scoring_pairs(requested_words)
            elif self.method == 2:
                mcps = weighted_top_pairs(requested_words)

            if not mcps:
                break

            removed_this_pass = 0

            for mcp in mcps:
                words = mcp.split()
                new_ngrams[words[0]] = mcp
                wset = set(words)
                for w in words:
                    ngram_lookup.setdefault(w, mcp)
                    for doc_id in reverse_doc_id_mathcing[w]:
                        ddict = docs_with_scores[doc_id]
                        # try to pop both words; += 1 for each successful pop
                        for ww in wset:
                            if ddict.pop(ww, None) is not None:
                                removed_this_pass += 1
            pbar.update(removed_this_pass)
        pbar.close()


        # we randomly assign any 'stragglers'
        stragglers = set()
        for doc_id, score in docs_with_scores.items():
            top_keywords = sorted(score.keys(), key=lambda x: x, reverse=True)
            stragglers.update(top_keywords)

        logging.debug(f" {len(stragglers)} stragglers")

        for i in range(0, len(stragglers), self.n):
            new_ngrams[stragglers[i]] = " ".join(stragglers[i:i + self.n])
            ngram_lookup[stragglers[i]] = new_ngrams[stragglers[i]]

        del stragglers

        ngram_bm25 = BM25Search(
            index_name=self.index_name,
            hostname=self.hostname,
            initialize=self.initialize,
            number_of_shards=self.shards,
            retry_on_timeout=True,
            timeout=600
        )

        logging.info(f"Running elastic search on newly made {self.n}-grams")
        final_hits = ngram_bm25.search(corpus, new_ngrams, top_k, score_function)


        new_lookup = {}
        for key, hits in final_hits.items():  # ngramlookup has some dead values
            for word in tokenize(ngram_lookup[key]):
                new_lookup[word] = hits

        del final_hits

        json.dump(new_lookup, open(f"{self.index_name}_{self.n}-gram_corpus_k@{top_k}.json", "w"))

        logging.info("Tokenising all documents...")

        # ---------- (1)  pre‑tokenise every document once ----------
        tokenised_docs = {doc_id: tokenize(doc_to_text(doc))
                          for doc_id, doc in original_corpus.items()}

        # ---------- (2)  build the final results container ----------
        final_res: dict[str, dict[str, float]] = {}
        track_qid_raw = {}

        # ---------- (3)  loop over queries exactly as before ----------
        for qid, query_text in tqdm(original_queries.items(), desc="Mapping n-grams to docs"):
            tokens = tokenize(query_text)

            # -- (3a) gather the per‑query document IDs from your lookup --
            doc_ids = set()  # no duplicates
            for tok in tokens:
                hits = new_lookup.get(tok)
                if hits is None:
                    logging.debug(f"missing word: {qid}, {tok}")
                    continue
                doc_ids.update(hits.keys())

            if not doc_ids:  # nothing to rank
                final_res[qid] = {}
                continue

            # -- (3b) build the *independent* corpus for this query --
            doc_ids_list = list(doc_ids)  # stable order
            track_qid_raw[qid] = doc_ids_list # dump these out before doing the BM25
            per_query_docs = [tokenised_docs[d] for d in doc_ids_list]

            # -- (3c) run a local BM25 over that slice only --
            bm25 = BM25Okapi(per_query_docs)  # in‑memory index
            scores = bm25.get_scores(tokens)  # 1 score per doc

            # -- (3d) keep the top‑k and store in BEIR format --
            top_idx = np.argsort(scores)[::-1][:top_k]
            final_res[qid] = {doc_ids_list[i]: float(scores[i])
                              for i in top_idx}
        json.dump(track_qid_raw, open(f"{self.index_name}_{self.n}-gram_results_per_qid_k@{top_k}.json", "w"))
        json.dump(final_res, open(f"{self.index_name}_{self.n}-gram_final_result_k@{top_k}.json", "w"))
        print(f"Finished running {self.n}-gram on k={top_k}")
        return final_res

        # # results is a mapping from qid to a dict of doc ids and their scores
        # final_res: dict[str, dict[str, float]] = {}
        #
        # for qid, query_text in tqdm(original_queries.items()):
        #     new_corpus = {}
        #     tokens = tokenize(query_text)
        #
        #     for token in tokens:
        #         results = new_lookup.get(token)
        #
        #         if results is None: # this word doesn't appear in our corpus
        #             logging.info(f"missing word: {qid}, {token}")
        #             continue
        #
        #         for doc_id, score in results.items():
        #             doc_items = original_corpus[doc_id]
        #             new_corpus[doc_id] = doc_items
        #
        #
        #     final_bm25 = BM25Search(
        #         index_name=self.index_name,
        #         hostname=self.hostname,
        #         initialize=self.initialize,
        #         number_of_shards=self.shards,
        #         retry_on_timeout=True,
        #         timeout=600
        #     )
        #
        #     with silence_stderr():  # hides *all* stderr output
        #         temp_results = final_bm25.search(new_corpus, {qid: query_text}, top_k, score_function)
        #
        #     for key, val in temp_results.items():
        #         final_res[key] = val
        #
        # return final_res




def main():

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    # datasets to test are scifact, nq, hotpotqa, quora, trec-covid, arguana
    # dataset = "quora"
    # dataset = "arguana"
    # dataset = "cqadupstack"
    # dataset = "hotpotqa"
    dataset = "scifact"
    # dataset = "nq"
    # this one is still rather slow, but unfortunately is the best...
    # dataset = "trec-covid"
    # dataset = "msmarco"
    # this one is the fastest (but both perform too well on this!)
    # dataset = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    # out_dir = os.path.join(pathlib.Path(__file__).parent, "datasets")
    out_dir = "/home/yelnat/Nextcloud/10TB-STHDD/datasets"
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

    print("======================= RESULTS FOR basic bm25 =======================")

    model = RegularBM25()

    retriever = EvaluateRetrieval(model, k_values=[10, 100])
    results = retriever.retrieve(corpus, queries)

    logging.info(f"Evaluation for k in {retriever.k_values}")
    ndcg, map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    mrr = retriever.evaluate_custom(qrels, results, k_values=retriever.k_values, metric="mrr")

    print(f"NDCG@{retriever.k_values}    : {ndcg}")
    print(f"MAP@{retriever.k_values}     : {map}")
    print(f"Recall@{retriever.k_values}  : {recall}")
    print(f"Precision@{retriever.k_values}: {precision}")
    print(f"MRR@{retriever.k_values}     : {mrr}")

    for i in [1, 2]:

        print("======================= RESULTS FOR n = {i} =======================".format(i=i))

        #model = ngramBM25Retriever(n=i)
        model = ngramBM25Retriever_freq(n=i, frequency=5 * i, method=2, index_name=dataset.lower())
        # model = RegularBM25()

        retriever = EvaluateRetrieval(model, k_values=[10, 100, 1000, 2000, 3000])
        results = retriever.retrieve(corpus, queries)

        logging.info(f"Evaluation for k in {retriever.k_values}")
        ndcg, map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        mrr = retriever.evaluate_custom(qrels, results, k_values=retriever.k_values, metric="mrr")

        print(f"NDCG@{retriever.k_values}    : {ndcg}")
        print(f"MAP@{retriever.k_values}     : {map}")
        print(f"Recall@{retriever.k_values}  : {recall}")
        print(f"Precision@{retriever.k_values}: {precision}")
        print(f"MRR@{retriever.k_values}     : {mrr}")



def doc_to_text(doc) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):                # BEIR standard
        return f"{doc.get('title', '')} {doc.get('text', '')}"
    return str(doc)

if __name__ == "__main__":
    main()


