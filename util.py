import hashlib
import struct
from typing import List, Set, Tuple
from tqdm import tqdm

def get_hash(s: str, n: int) -> int:
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
    bin_choices = []

    for choice in range(d):
        index = get_hash(word, choice) % max_bins
        
        overlap = sum(1 for idx in orig_results[index] if idx in document_ids)
#         print(orig_results[index], overlap, document_ids)
        overlap -= len(document_ids)

        if overlap >= (2**64 - 111):
            raise ValueError(f"Overlap was negative? {overlap}")

        bin_size = len(orig_results[index])

#         print(f"Got index {index}, overlap: {overlap}, k: {search_results_len}, bin size: {bin_size}")
        
        bin_choices.append((index, bin_size, overlap))
    
    return bin_choices

def remove_min_overlap(bins: List[Tuple[int, int, int]], count: int) -> List[Tuple[int, int, int]]:
    bins.sort(key=lambda x: x[2])  # Sort by overlap (ascending)
    return bins[count:]  # Remove first `count` elements

def remove_max_load(bins: List[Tuple[int, int, int]], count: int) -> List[Tuple[int, int, int]]:
    bins.sort(key=lambda x: x[2], reverse=True)  # Sort by overlap (descending)
    return bins[count:]  # Remove first `count` elements

import copy

def top_k_bins(retriever, config):
    k, d, max_bins, filter_k, max_load_factor, min_overlap_factor = [*config.values()]
    print(k, d, max_bins, filter_k, max_load_factor, min_overlap_factor)
    results = [[] for _ in range(max_bins)]
    orig_results = [[] for _ in range(max_bins)]
    archived_results = []
    total_overlap = 0
    keywords_with_overlap = 0
    alphabet = retriever.vocab_dict
    c = 0
    for word, word_token in tqdm(alphabet.items()):
        
#         print(word_token, word)
        search_results, scores = retriever.retrieve([[word_token]], k = k)
        search_results = [result for i, result in enumerate(search_results[0]) if scores[0][i] != 0.0]
        # convert result to document ids
        document_ids = list(set(search_results))#copy.deepcopy(search_results)
#         print(document_ids)
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
        for choice in bin_choices[:1]:
            if max_overlap < choice[2]:
                max_overlap = choice[2]
            
            results[choice[0]].extend(document_ids)
        
        total_overlap += max_overlap
        
        if max_overlap > 0:
            keywords_with_overlap += 1
    return {
        "num_bins": max_bins,
        "k": k,
        "d": d,
        "removed_items": total_overlap,
        "total_items": sum([len(s) for s in results]),
        "average_load_per_bin": sum([len(s) for s in results]) / len(results),
        "keywords_with_overlap": keywords_with_overlap
    }, results
        
    