import logging
from tkinter.scrolledtext import example

from datasets import load_dataset

from BM25_PIR.experiments import do_search
from BM25_PIR.dataloader import load_nyt, process_text
import bm25s
import time
from Stemmer import Stemmer

from BM25_sanity_check.experiments import classic_BM25, tokenised_BM25


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

def bm25_sanity_experiment():
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

    logging.info("Starting classic BM25 test")
    token_res = tokenised_BM25(squad_dataset, 20, retriever, stemmer, corpus_list, 5000)
    classic_res = classic_BM25(squad_dataset, 20, retriever, stemmer, corpus_list, 5000)

    average_classic = sum(classic_res) / len(classic_res)
    print(average_classic)

    average_tokens = sum(token_res) / len(token_res)
    print(average_tokens)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bm25_sanity_experiment()

