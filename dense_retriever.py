import json
from typing import List, Dict

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the dense retriever with a pre-trained sentence transformer model.

        Args:
            model_name: The name of the sentence transformer model to use.
                        Default is 'all-MiniLM-L6-v2' which is one of the few models I can run locally and is also popular
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.document_embeddings = None

    def load_jsonl(self, filepath: str) -> None:
        """
        Load documents from a JSONL file where each line contains a JSON with a 'text' field.

        Args:
            filepath: Path to the JSONL file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.documents = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(self.documents)} documents")

    def encode_documents(self, batch_size: int = 32) -> None:
        """
        Encode all documents into dense vector embeddings.

        Args:
            batch_size: Batch size for encoding
        """
        texts = [doc['text'] for doc in self.documents]

        # Use batching and show progress bar for large document collections
        print("Encoding documents...")
        self.document_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True)
            self.document_embeddings.append(batch_embeddings)

        # Concatenate all batches
        self.document_embeddings = torch.cat(self.document_embeddings)
        print(f"Created embeddings with shape: {self.document_embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 10) -> (List[int], List[float]):
        """
        Retrieve the top_k most relevant documents for a query.

        Args:
            query: The query text
            top_k: Number of top documents to retrieve

        Returns:
            List of dictionaries with document content and similarity score
        """
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute similarity scores (per word)
        cos_scores = self.compute_similarity(query_embedding)

        # I think this is how to do this, maybe.
        top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores))).indices

        results = []
        scores = []
        # this should only be one idx
        for idx in top_indices:
            results.append(idx.item())
            scores.append(cos_scores[idx].item())

        return results, scores

    def compute_similarity(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between query and all documents.

        Args:
            query_embedding: The query embedding tensor

        Returns:
            Tensor of similarity scores
        """
        # Compute cosine similarity
        query_embedding = query_embedding / query_embedding.norm()
        document_embeddings_normalized = self.document_embeddings / self.document_embeddings.norm(dim=1, keepdim=True)

        return torch.matmul(document_embeddings_normalized, query_embedding)