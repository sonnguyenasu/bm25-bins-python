from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import os
from config_classes import dataclass
from tabulate import tabulate


from util import Metadata


@dataclass
class ExperimentResult:
    """Used for displaying a table with tabulate"""
    name: str
    bins: int
    removed: int
    total: int
    avg_load: int
    keywords: int


def fullness_histogram(
        histogram: list[set[int]],
        sorted: bool,
        title: str,
        granularity: int
) -> None:
    """
    Takes in a list of sets and plots a histogram showing the number of items in each bin

    Args:
        histogram: A list of 'bins', each bin should be a set of document IDs
        sorted: If set to true, puts the largest bin on the left of the histogram
        title: Title for the histogram
        granularity: Number of groups to consolidate bins into
    """
    # Make a copy to avoid modifying the original
    histogram = histogram.copy()

    if sorted:
        histogram.sort(key=lambda b: len(b), reverse=True)

    bin_counts = [(idx, len(s)) for idx, s in enumerate(histogram)]

    # Consolidate bins into groups based on granularity
    target_bins = granularity
    bins_per_group = int(np.ceil(len(bin_counts) / target_bins))

    consolidated_bins = []
    for idx, chunk in enumerate([bin_counts[i:i + bins_per_group]
                                 for i in range(0, len(bin_counts), bins_per_group)]):
        total = sum(count for _, count in chunk)
        consolidated_bins.append((idx, total))

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 7.5))

    x_values = [idx for idx, _ in consolidated_bins]
    y_values = [count for _, count in consolidated_bins]

    max_count = max(y_values) if y_values else 0
    y_max = int(max_count * 1.1)  # Add 10% margin

    ax.bar(x_values, y_values, color='red', width=0.8)

    ax.set_xlim(-0.5, len(consolidated_bins) - 0.5)
    ax.set_ylim(0, y_max)

    ax.set_xlabel('Bin Number')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=16)

    # Remove grid lines
    ax.grid(False)

    # Create directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Save the figure
    output = f"figures/{title}_histogram.png"
    plt.savefig(output)
    plt.close(fig)

    print(f"Histogram saved to {output}")


def analyze_doc_id_frequency(results: dict[str, set[int]]) -> Counter[int]:
    """
    Analyzes and visualizes the frequency of document IDs across all search results.

    Args:
        results: Dictionary mapping words to sets of matching document IDs (output from top_k function)

    Returns:
        The counter that counted the number of documents across all search results
    """
    # Change from set into list and flatten. Should make it easier to count
    all_doc_ids = []
    for word, doc_id_set in results.items():
        all_doc_ids.extend(list(doc_id_set))

    # Count frequency of each document ID
    doc_id_counter = Counter(all_doc_ids)

    # Sort by document ID for clearer visualization
    sorted_items = sorted(doc_id_counter.items())
    doc_ids, frequencies = zip(*sorted_items) if sorted_items else ([], [])

    # Calculate statistics
    total_docs = len(doc_id_counter)
    max_frequency = max(frequencies) if frequencies else 0
    avg_frequency = sum(frequencies) / len(frequencies) if frequencies else 0

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(doc_ids)), frequencies, alpha=0.7)
    plt.xlabel('Document ID Index (sorted by ID)')
    plt.ylabel('Frequency')
    plt.title(f'Document ID Frequency Distribution\n(Total unique docs: {total_docs}, Avg freq: {avg_frequency:.2f})')

    # Add a histogram of the frequencies themselves
    plt.subplot(1, 2, 2)
    bins = np.arange(1, max_frequency + 2) - 0.5
    plt.hist(frequencies, bins=bins, alpha=0.7)
    plt.xlabel('Frequency')
    plt.ylabel('Count')
    plt.title('Histogram of Document ID Frequencies')
    plt.savefig("figures/doc_id_frequency.png")

    plt.tight_layout()
    plt.show()

    # Print some additional statistics
    print(f"Total unique document IDs: {total_docs}")
    print(f"Most common document ID: {doc_id_counter.most_common(1)[0][0]} (appears {max_frequency} times)")
    print(f"Average appearance per document ID: {avg_frequency:.2f}")

    return doc_id_counter

def overlap_histogram(
        histogram1: dict[str, set[int]],
        histogram2: dict[str, set[int]],
        sorted: bool,
        title: str,
        granularity: int
) -> None:
    """
    Takes in two dictionaries mapping strings to sets of ints and plots a histogram
    showing the overlap between them

    Args:
        histogram1: First dictionary where keys are strings and values are sets of document IDs
        histogram2: Second dictionary where keys are strings and values are sets of document IDs
        sorted: If set to true, puts the bins with largest overlap on the left
        title: Title for the histogram
        granularity: Number of groups to consolidate bins into
    """


    # Get all unique keys from both dictionaries
    all_keys = set(histogram1.keys()).union(histogram2.keys())

    # Calculate overlap for each key
    overlaps = []
    total_overlap = 0

    for key in all_keys:
        set1 = histogram1.get(key, set())
        set2 = histogram2.get(key, set())
        overlap = len(set1.intersection(set2))
        overlaps.append((key, overlap))
        total_overlap += overlap

    # Sort by overlap size if requested
    if sorted:
        overlaps.sort(key=lambda x: x[1], reverse=True)

    # Consolidate bins into groups based on granularity
    target_bins = min(granularity, len(overlaps))
    bins_per_group = int(np.ceil(len(overlaps) / target_bins))
    consolidated_bins = []

    for idx, chunk in enumerate([overlaps[i:i + bins_per_group]
                                 for i in range(0, len(overlaps), bins_per_group)]):
        total = sum(count for _, count in chunk)
        # Store the keys in this group for potential labeling
        keys_in_group = [key for key, _ in chunk]
        consolidated_bins.append((idx, total, keys_in_group))

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 7.5))
    x_values = [idx for idx, _, _ in consolidated_bins]
    y_values = [count for _, count, _ in consolidated_bins]

    max_count = max(y_values) if y_values else 0
    y_max = int(max_count * 1.1)  # Add 10% margin

    ax.bar(x_values, y_values, color='purple', width=0.8)
    ax.set_xlim(-0.5, len(consolidated_bins) - 0.5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('Bin Number')
    ax.set_ylabel('Overlap Count')
    ax.set_title(f"{title}\nTotal Overlap: {total_overlap} items", fontsize=16)

    # Remove grid lines
    ax.grid(False)

    # Create directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Save the figure
    output = f"figures/{title}_overlap_histogram.png"
    plt.savefig(output)
    plt.close(fig)

    # Print the total overlap count
    print(f"Total overlap between histograms: {total_overlap} items")
    print(f"Overlap histogram saved to {output}")

    return total_overlap


def print_table(
        experiment_names: list[str],
        metadata_vec: list[Metadata]
) -> None:
    """
    Prints a table to the terminal for displaying experiment metadata

    Args:
        experiment_names: Names for each experiment
        metadata_vec: Metadata objects containing experiment statistics
    """
    results = []

    for name, meta in zip(experiment_names, metadata_vec):
        results.append(ExperimentResult(
            name=name,
            bins=meta.num_bins,
            removed=meta.removed_items,
            total=meta.total_items,
            avg_load=meta.average_load_per_bin,
            keywords=meta.keywords_with_overlap
        ))

    # Create table headers matching the Rust version
    headers = [
        "Experiment Name",
        "# Bins",
        "Items Removed",
        "Total Items",
        "Avg Load",
        "Keywords w/Overlap"
    ]

    # Extract data from results
    table_data = [[
        result.name,
        result.bins,
        result.removed,
        result.total,
        result.avg_load,
        result.keywords
    ] for result in results]

    # Print table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
