import hashlib
import struct
from BM25_PIR.config_classes import dataclass, Metadata, Config
from typing import Tuple, Union, Any, Literal
from typing import List, Set

from bm25s import BM25
from tqdm import tqdm
import logging

import logging
from typing import Dict, Set, List, Any, Optional
from collections import defaultdict
from tqdm import tqdm  # For progress bar


def top_k(
        k: int,
        search_engine: BM25,
        filter_k: int,
) -> Dict[str, Set[int]]:
    """
    Performs top-k search for each word in the alphabet and filters results.
    Doesn't do any choice hashing, just returns top-k results.
    Theoretical return size is O(k * alphabet), i.e. each bin has k results.

    Args:
        k: Number of results to retrieve per word (the k in top-k)
        search_engine: Search engine to query
        filter_k: Minimum number of results required to keep a word
                 (e.g., if filter_k is 2, results with only 1 match will be discarded)

    Returns:
        Dict mapping words to sets of matching document IDs
    """
    results = {}
    alphabet = search_engine.vocab_dict


    # Track duplicates for logging
    counting_duplicates = defaultdict(int)
    num_items = 0

    for word, word_token in tqdm(alphabet.items()):
        # Get search results for this word
        search_results, scores = search_engine.retrieve([[word_token]], k = k)
        search_results = [result for i, result in enumerate(search_results[0]) if scores[0][i] != 0.0]

        document_ids = list(set(search_results))  # copy.deepcopy(search_results)
        logging.debug(document_ids)

        # Filter out words with too few results
        if len(search_results) < filter_k:
            continue

        doc_ids = list(set(search_results))
        # Process the results
        for doc_id in doc_ids:
            # Initialize the set for this word if it doesn't exist
            if word not in results:
                results[word] = set()

            results[word].add(doc_id)

            # Count items and track duplicates
            num_items += 1
            if doc_id in counting_duplicates:
                counting_duplicates[doc_id] += 1
            else:
                counting_duplicates[doc_id] = 0


    # Log summary information
    total_duplicates = sum(counting_duplicates.values())
    logging.info(
        f"Top-K done without d-choice. Total number of duplicates: {total_duplicates}, "
        f"total items in bins: {num_items}"
    )

    if results:  # Avoid division by zero
        avg_items_per_bin = sum(len(item_set) for item_set in results.values()) / len(results)
        logging.info(f"The average number of items in bins is {avg_items_per_bin}")

    return results

def get_hash(s: str, n: int) -> int:
    """
    Hashes a string and integer using SHA256
    :param s: A string to use as input to SHA256
    :param n: A integer to use as input to SHA256
    :return: An int of the first 8 bytes of the SHA256 hash
    """
    hasher = hashlib.sha256()
    hasher.update(s.encode('utf-8'))
    hasher.update(str(n).encode('utf-8'))
    result = hasher.digest()
    
    # Convert first 8 bytes of SHA-256 hash to int (big-endian)
    return struct.unpack('>Q', result[:8])[0]

def get_bins(word: str,
             d: int, max_bins: int,
             document_ids: Set[int],
             orig_results: List[List[int]],
             search_results_len: int) -> List[Tuple[int, int, int]]:
    """
    Returns a list of: Index that the item should be put in, how full that bin is already and the overlap that
     would occur in that bin.
    :param word: The word to determine where index
    :param d: the number of choices in d-choice hashing
    :param max_bins: The maximum number of bins, to modulo the SHA result by
    :param document_ids: The unique document IDs that were found by the BM25 scorer
    :param orig_results: The original results, in which we blindly placed items into bins. We used this to determine
    what the overlap and load of the bins would be naturally (i.e. so if it's the first word we're deciding for we still
    have a good estimation of where the min overlap will be)
    :param search_results_len: Used for loggin purposes
    :return: a list of 'd' tuples in the form: (index, load, overlap)
    """
    bin_choices = []

    # Cycle through our d choices
    for choice in range(d):
        index = get_hash(word, choice) % max_bins
        
        overlap = sum(1 for idx in orig_results[index] if idx in document_ids)
        logging.debug(orig_results[index], overlap, document_ids)
        overlap -= len(document_ids)

        if overlap >= (2**64 - 111):
            raise ValueError(f"Overlap was negative? {overlap}")

        bin_size = len(orig_results[index])

        logging.debug(f"Got index {index}, overlap: {overlap}, k: {search_results_len}, bin size: {bin_size}")
        
        bin_choices.append((index, bin_size, overlap))
    
    return bin_choices

def remove_min_overlap(bins: List[Tuple[int, int, int]], count: int) -> List[Tuple[int, int, int]]:
    """
    Pop off the minimum overlap bins and return the remaining bins
    :param bins: A list of tuples, expected to be in the form (index, load, overlap)
    :param count: The number of bins to remove
    :return: The bins list missing 'count' bins
    """
    bins.sort(key=lambda x: x[2])  # Sort by overlap (ascending)
    return bins[count:]  # Remove first `count` elements

def remove_max_load(bins: List[Tuple[int, int, int]], count: int) -> List[Tuple[int, int, int]]:
    """
    Pop off the maximum load bins and return the remaining bins
    :param bins: A list of tuples, expected to be in the form (index, load, overlap)
    :param count: The number of bins to remove
    :return: The bins list missing 'count' bins
    """
    bins.sort(key=lambda x: x[2], reverse=True)  # Sort by overlap (descending)
    return bins[count:]  # Remove first `count` elements

import copy

def top_k_bins(retriever: BM25,
               config: Config) -> tuple[Metadata, list[set[int]]]:
    """
    Takes in the bm25s retriever object and extracts the alphabet from it, then iterates over the entire keyspace and maps
    the items into the desired number of bins. See the config object to determine which values are to be unpacked.
    :param retriever: the BM25s retriever object
    :param config: configurable parameters, like k, filter_k, max_bins... etc (This should be changed to a class or
    some such for ease of use
    :return: A metadata object with various interesting metadata inside and a list of bins with document IDs inside
    """
    k = config.k
    d = config.d
    max_bins = config.max_bins
    filter_k = config.filter_k
    max_load_factor = config.max_load_factor
    min_overlap_factor = config.min_overlap_factor
    logging.info(f"{k}, {d}, {max_bins}, {filter_k}, {max_load_factor}, {min_overlap_factor}")
    results = [set() for _ in range(max_bins)]
    orig_results = [[] for _ in range(max_bins)]
    archived_results = []
    total_overlap = 0
    keywords_with_overlap = 0
    alphabet = retriever.vocab_dict
    c = 0
    for word, word_token in tqdm(alphabet.items()):
        
        logging.debug(f"{word_token}, {word}")
        search_results, scores = retriever.retrieve([[word_token]], k = k)
        search_results = [result for i, result in enumerate(search_results[0]) if scores[0][i] != 0.0]
        # convert result to document ids
        document_ids = list(set(search_results))#copy.deepcopy(search_results)
        logging.debug(document_ids)
        # skip words with too few results
        if len(search_results) < filter_k:
            continue
        
        # add document id into the correspondng bins
        for choice in range(d):
            index = get_hash(word, choice) % max_bins
            orig_results[index].extend(copy.deepcopy(document_ids))
        archived_results.append((word, word_token, search_results))
#         print(orig_results[5541])
#         c+= 1
#         if c >= 10:
#             break
    
    for (word, word_token, search_results) in tqdm(archived_results):
        document_ids = list(set(search_results))
        bin_choices = get_bins(word, d, max_bins, document_ids, orig_results, len(search_results))
        bin_choices = remove_min_overlap(bin_choices, min_overlap_factor)
        bin_choices = remove_max_load(bin_choices, max_load_factor)
        
        max_overlap = 0
        # Store in every bin that is left over, i.e. if you set to remove 1 max overlap and 1 min load but have d = 4,
        # then you will end up with duplicated results in two bins
        for choice in bin_choices:
            # If there are multiple bins then it's difficult to get a good metric for how much overlap is saved. I just
            # take the max because I can't be bothered thinking about it, we can just work it out afterwards by looking
            # at how many items are in the bins anyway
            if max_overlap < choice[2]:
                max_overlap = choice[2]
            
            results[choice[0]].update(document_ids)
        
        total_overlap += max_overlap
        
        if max_overlap > 0:
            keywords_with_overlap += 1

    metadata = Metadata(
        num_bins=max_bins,
        k=k,
        d=d,
        removed_items=total_overlap,
        total_items=sum(len(s) for s in results),
        average_load_per_bin=sum(len(s) for s in results) / len(results) if results else 0,
        keywords_with_overlap=keywords_with_overlap
    )

    return metadata, results
        
    