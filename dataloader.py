import pandas as pd
from typing import List
from Stemmer import Stemmer
import bm25s

def load_nyt(filepath: str) -> List[str]:
    corpus = pd.read_json(path_or_buf=filepath, lines=True)
    return corpus['text'].to_list()

def process_text(texts: List[str], stemmer: Stemmer):
    return bm25s.tokenize(texts, stemmer)