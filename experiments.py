import logging
from typing import Set, List, Dict, Any
from config_classes import dataclass, Config

from bm25s import BM25

# Import the modules we've defined in the previous artifacts
# Assuming these functions are defined in their respective modules
from plotter import fullness_histogram, print_table
from util import top_k, top_k_bins, Metadata


def do_search(
        k: int,
        filter_k: int,
        search: BM25
):
    """
    Execute search and visualization routines on the given corpus.

    Args:
        k: Number of top elements to consider
        filter_k: Filter parameter
        :param search:
    """


    logging.info(
        f"The total number of files is {0xDEADBEEF} and the alphabet size is {len(search.vocab_dict.keys())}"
    )


    # Get top_k results
    top_k_res = top_k(k, search, filter_k)
    logging.info("Top K Done")

    # Plot histogram for top k results
    fullness_histogram(
        list(top_k_res.values()),
        True,
        "Top K (No bins)",
        len(top_k_res.values())
    )

    max_bins = len(top_k_res.values()) // 10

    # Create default config
    config = Config(
        k=10,
        d=1,
        max_bins=max_bins,
        filter_k=filter_k,
        max_load_factor=0,
        min_overlap_factor=0,
        save_result=True
    )

    # 1-choice bins
    no_choice_bins = top_k_bins(search, config)
    fullness_histogram(
        no_choice_bins[1],
        True,
        f"Top K 1-choice {max_bins}-bins",
        max_bins
    )

    # 2-choice bins
    config.d = 2
    config.min_overlap_factor = 1
    two_choice_bins = top_k_bins(search, config)
    fullness_histogram(
        two_choice_bins[1],
        True,
        f"Top K 2-choice {max_bins}-bins",
        max_bins
    )

    # 3-choice bins
    config.d = 3
    config.min_overlap_factor = 2
    three_choice_bins = top_k_bins(search, config)
    fullness_histogram(
        three_choice_bins[1],
        True,
        f"Top K 3-choice {max_bins}-bins",
        max_bins
    )

    # 3-choice bins with 1 max load bin removed
    config.d = 3
    config.min_overlap_factor = 1
    config.max_load_factor = 1
    three_choice_bins_remove_one = top_k_bins(search, config)
    fullness_histogram(
        three_choice_bins_remove_one[1],
        True,
        f"3-choice, {max_bins}-bins and 1 max-load bin removed",
        max_bins
    )

    # 2-choice bins minimizing load
    config.d = 2
    config.min_overlap_factor = 0
    config.max_load_factor = 1
    two_choice_bins_max_load = top_k_bins(search, config)
    fullness_histogram(
        two_choice_bins_max_load[1],
        True,
        f"Top K 2-choice {max_bins}-bins, minimising load",
        max_bins
    )

    # Commented out in original:
    # # 100-choice with 10 max load bins removed
    # config.d = 100
    # config.max_load_factor = 10
    # config.min_overlap_factor = 89
    # hundred_choice_ten_max_load = top_k_bins(search, config)
    # fullness_histogram(
    #     hundred_choice_ten_max_load[1],
    #     True,
    #     f"Top K 100-choice {max_bins}-bins, remove 10 max load bins",
    #     max_bins
    # )

    # 4-choice with min overlap and max load removal
    config.d = 4
    config.min_overlap_factor = 1
    config.max_load_factor = 1
    four_choice_min_overlap_max_overlap = top_k_bins(search, config)
    fullness_histogram(
        four_choice_min_overlap_max_overlap[1],
        True,
        f"4-choice {max_bins}-bins, remove 1 min overlap, 1 max load",
        max_bins
    )

    # Create format strings and results for the table
    format_strings = [
        f"Naive 1-1 mapping with {len(top_k_res.values())}-bins",
        f"1-choice {max_bins}-bins",
        f"2-choice {max_bins}-bins",
        f"3-choice {max_bins}-bins",
        f"3-choice, {max_bins}-bins and 1 max-load bin removed",
        f"2-choice {max_bins}-bins, minimising load",
        # f"100-choice {max_bins}-bins, remove 10 max load bins",
        f"4-choice {max_bins}-bins, remove 1 min overlap, 1 max load"
    ]

    # Create metadata for top_k
    total_items = sum(len(s) for s in top_k_res.values())
    top_k_meta = Metadata(
        num_bins=len(top_k_res.keys()),
        k=k,
        d=1,
        removed_items=0,
        total_items=total_items,
        average_load_per_bin=total_items / len(top_k_res.values()) if top_k_res.values() else 0,
        keywords_with_overlap=0
    )

    # Collect all results
    results = [
        top_k_meta,
        no_choice_bins[0],
        two_choice_bins[0],
        three_choice_bins[0],
        three_choice_bins_remove_one[0],
        two_choice_bins_max_load[0],
        # hundred_choice_ten_max_load[0],
        four_choice_min_overlap_max_overlap[0]
    ]

    # Print the table
    print_table(format_strings, results)
