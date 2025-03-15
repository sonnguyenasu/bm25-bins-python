from dataclasses import dataclass


@dataclass
class Config:
    """Configuration parameters for top_k_bins function"""
    k: int
    d: int
    max_bins: int
    filter_k: int
    max_load_factor: int
    min_overlap_factor: int
    save_result: bool = False  # Default value matching the Rust struct

@dataclass
class Metadata:
    """Metadata class to match the Rust implementation"""
    num_bins: int
    k: int
    d: int
    removed_items: int
    total_items: int
    average_load_per_bin: int
    keywords_with_overlap: int