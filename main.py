import itertools
import logging
import os
from tkinter.scrolledtext import example

from datasets import load_dataset

from BM25_PIR.experiments import do_search
from BM25_PIR.dataloader import load_nyt, process_text
import bm25s
import time
from Stemmer import Stemmer

from collections import defaultdict
import ir_datasets

from BM25_sanity_check.experiments import classic_BM25, tokenised_BM25
from BM25_sanity_check.make_embeddings import embed_and_export_fvecs


def bm25_d_choice_experiment():
    print("Loading dataset")
    start = time.time()
    corpus = load_nyt("BM25_PIR/nyt_processed_regex.jsonl")
    print("Dataset loaded in {} seconds".format(time.time() - start))
    print("Indexing")
    retriever = bm25s.BM25(backend="numba")
    stemmer = Stemmer("english")
    corpus_token = process_text(corpus, stemmer)
    retriever.index(corpus_token)
    print("Indexing complete")


    do_search(10, 5, retriever)

    # config = Config(
    #     k=10,
    #     d=3,
    #     max_bins=5895,
    #     filter_k=5,
    #     max_load_factor=0,
    #     min_overlap_factor=0
    # )
    # metadata, results = top_k_bins(retriever,
    #   config)

def bm25_sanity_experiment_squad():
    logging.info("Loading SQUAD dataset")

    squad_dataset = load_dataset("rajpurkar/squad_v2", split='validation')

    retriever = bm25s.BM25(backend="numba")

    # Initialize an empty corpus list for sentences and a dictionary for question-to-answers mapping
    corpus = set()
    question_answers = {}
    stemmer = Stemmer("english")

    # Loop over each example in the training dataset
    for example in squad_dataset:

        # Get the question and its answers (answers are stored under the "answers" key as a dict)
        question = example['question']

        full_answers = example['context']
        sentences = full_answers.split(".")  # split by sentences

        for sentence in sentences:
            corpus.add(sentence)


        if question in question_answers:
            continue
        else:
            question_answers[question] = [example['answers'], example['context']]

    corpus_list = list(corpus)
    corpus_token = process_text(corpus_list, stemmer)
    retriever.index(corpus_token)

    logging.info(f"Total sentences extracted: {len(corpus)}")
    logging.info(f"Total unique questions:{len(question_answers)}")

    # Export th embeddings as vectors
    logging.info("Exporting embeddings to fvecs files...")
    questions = list(question_answers.keys())
    embed_and_export_fvecs(
        corpus_sentences=corpus_list,
        questions=questions,
        out_prefix="squad_val",
    )

    logging.info("Starting classic BM25 test")
    token_res = tokenised_BM25(squad_dataset, 10, retriever, stemmer, corpus_list, 500)
    classic_res = classic_BM25(squad_dataset, 10, retriever, stemmer, corpus_list, 500)

    average_classic = sum(classic_res) / len(classic_res)
    print(average_classic)

    average_tokens = sum(token_res) / len(token_res)
    print(average_tokens)


def bm25_sanity_experiment_marco_dev():
    logging.info("Loading MS MARCO Passage Ranking /dev dataset using ir_datasets")
    # Load the development split from the MS MARCO Passage Ranking dataset.
    dataset = ir_datasets.load("msmarco-passage/dev")

    # Limit the number of queries for this experiment (to avoid long processing times).
    sample_size = 500  # Adjust this number if needed.
    queries = {
        query.query_id: query.text
        for query in itertools.islice(dataset.queries_iter(), sample_size)
    }

    # Build a dictionary of documents (passages)
    docs = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

    # Build relevance mappings: map query_id to a list of (doc_id, relevance) tuples
    qrels = defaultdict(list)
    for qrel in dataset.qrels_iter():
        # Restrict to only those queries in our sample.
        if qrel.query_id in queries:
            qrels[qrel.query_id].append((qrel.doc_id, qrel.relevance))

    # Build corpus and map queries to their associated answers and contexts.
    corpus = set()
    question_answers = {}
    # For each query in our sample:
    for query_id, query_text in queries.items():
        # Get all passage documents (with their relevance scores) for this query.
        doc_list = qrels.get(query_id, [])
        # For this experiment, consider passages with relevance > 0 as relevant.
        relevant_docs = [
            docs[doc_id] for (doc_id, relevance) in doc_list
            if relevance > 0 and doc_id in docs
        ]
        # Build a context string by joining all relevant passages.
        context = " ".join(relevant_docs) if relevant_docs else ""
        # Extract individual sentences from each passage and add to the corpus.
        for doc in relevant_docs:
            # A simple sentence splitter; for better quality consider using NLTK or spaCy.
            for sentence in doc.split("."):
                sentence = sentence.strip()
                if sentence:
                    corpus.add(sentence)
        # Map this query to its answers (the list of relevant passages) and full context.
        question_answers[query_text] = [relevant_docs, context]

    corpus_list = list(corpus)
    logging.info(f"Total sentences extracted: {len(corpus)}")
    logging.info(f"Total unique queries processed: {len(question_answers)}")

    # Prepare BM25 retrieval: process the corpus and index it.
    stemmer = Stemmer("english")
    corpus_token = process_text(corpus_list, stemmer)
    retriever = bm25s.BM25(backend="numba")
    retriever.index(corpus_token)

    logging.info("Starting BM25 evaluation")

    token_res = tokenised_BM25(dataset, 10, retriever, stemmer, corpus_list, 500)
    classic_res = classic_BM25(dataset, 10, retriever, stemmer, corpus_list, 500)

    average_classic = sum(classic_res) / len(classic_res)
    print("Average Classic BM25 score:", average_classic)

    average_tokens = sum(token_res) / len(token_res)
    print("Average Tokenised BM25 score:", average_tokens)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bm25_sanity_experiment_marco_dev()
