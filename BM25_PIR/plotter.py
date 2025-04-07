import matplotlib.pyplot as plt
import numpy as np
import os
from BM25_PIR.config_classes import dataclass
from typing import List, Set, Tuple
from tabulate import tabulate

from BM25_PIR.util import Metadata


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
        histogram: List[Set[int]],
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
    os.makedirs('../figures', exist_ok=True)

    # Save the figure
    output = f"figures/{title}_histogram.png"
    plt.savefig(output)
    plt.close(fig)

    print(f"Histogram saved to {output}")


def print_table(
        experiment_names: List[str],
        metadata_vec: List[Metadata]
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
