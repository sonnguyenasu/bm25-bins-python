import bm25s
from Stemmer import Stemmer
from bm25s import BM25
from datasets import load_dataset, Dataset
from sklearn.metrics import recall_score

from BM25_PIR.dataloader import process_text


def classic_BM25(dataset: Dataset, k: int,
        search: BM25, stemmer: Stemmer, corpus: list[str],
        samples_to_test: int):
    """
    Takes in a dataset and then performs a standard BM25 search over the number of specified training samples.
    The BM25 should return k sentences, of which we need to check if the sentences are: contained within the correct answer
    and after where the correct answer is found. We compute the F1 score as follows: True positive is the number of
    correct sentences, regardless of where correct answer was (as long as it's in the training sample), the false
    positive is the number of wrong sentences and the false negatives are the number of sentences that should've been
    retrieved, but where missed.
    :param dataset: the hugginface dataset
    :param k: the number of items BM25 will retrieve
    :param search: the bm25 searcher
    :param samples_to_test: the number of samples to get from the dataset
    :return: a list of scores, each index relating to the index taken from the dataset
    """

    f1_scores = []

    for example in dataset.select(range(samples_to_test)):
        question = example['question']

        real_answers = example['context'].split('.')

        question_tokens = bm25s.tokenize(question, stemmer)

        search_results, scores = search.retrieve(question_tokens, k=k)

        real_results = []

        for result in search_results[0]:
            real_results.append(corpus[result])

        true_positives = 0
        false_positives = 0

        for i in range(len(real_results)):

            if real_results[i] in real_answers:
                true_positives += 1
            else:
                false_positives += 1

        false_negatives = k - true_positives

        f1_scores.append((2 * true_positives) / ((2 * true_positives) + false_positives + false_negatives))

    return f1_scores






def tokenised_BM25(dataset: Dataset, k: int,
        search: BM25, stemmer: Stemmer, corpus: list[str],
        samples_to_test: int):
    """
    Takes in a dataset and then performs a tokenised BM25 search over the number of specified training samples.
    The BM25 should return k * w sentences, where w is the number of qords in the query. THis is the same as classic BM25
    but searches one word at a time.
    :param dataset: the hugginface dataset
    :param k: the number of items BM25 will retrieve
    :param search: the bm25 searcher
    :param samples_to_test: the number of samples to get from the dataset
    :return: a list of scores, each index relating to the index taken from the dataset
    """


    f1_scores = []
    mrr_score = []

    for example in dataset.select(range(samples_to_test)):
        question = example['question']

        new_corpus = []

        for word in question.split(' '):

            question_tokens = bm25s.tokenize(word, stemmer)

            if not question_tokens.vocab:
                continue

            search_results, scores = search.retrieve(question_tokens, k=k)

            for result in search_results[0]:
                new_corpus.append(corpus[result])

        new_retriever = bm25s.BM25(backend="numba")
        corpus_token = process_text(new_corpus, stemmer)
        new_retriever.index(corpus_token)

        real_answers = example['context'].split('.')

        question_tokens = bm25s.tokenize(question, stemmer)

        search_results, scores = new_retriever.retrieve(question_tokens, k=k)

        real_results = []

        for result in search_results[0]:
            real_results.append(new_corpus[result])

        true_positives = 0
        false_positives = 0

        for i in range(len(real_results)):

            if real_results[i] in real_answers:
                true_positives += 1
            else:
                false_positives += 1

        false_negatives = k - true_positives

        f1_scores.append((2 * true_positives) / ((2 * true_positives) + false_positives + false_negatives))
        if true_positives > 0:
            mrr_score.append(1)

    return f1_scores, mrr_score


