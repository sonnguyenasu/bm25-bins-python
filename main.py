import logging

import torch

from dense_retriever import DenseRetriever
from experiments import do_dense_search
from dataloader import load_nyt, process_text
import bm25s
import time
from Stemmer import Stemmer



if __name__ == "__main__":

    filename = "./nyt_processed_regex.jsonl"
    print("Loading dataset")
    start = time.time()
    corpus = load_nyt(filename)
    print("Dataset loaded in {} seconds".format(time.time() - start))
    print("Indexing")
    retriever = bm25s.BM25(backend="numba")
    stemmer = Stemmer("english")
    corpus_token = process_text(corpus, stemmer)
    retriever.index(corpus_token)
    print("Indexing complete")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dense_retriever = DenseRetriever(filepath=filename, device=device)

    #do_bm25_search(10, 10, retriever)
    do_dense_search(10, 2, retriever, dense_retriever)



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

