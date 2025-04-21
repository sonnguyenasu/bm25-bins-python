
import json, pathlib, argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def write_fvecs(path: pathlib.Path, mat: np.ndarray) -> None:
    """
    Save a 2‑D float32 array to Faiss/SIFT .fvecs format:
      int32(dim)  +  dim*float32   for every row.
    """
    mat = np.asarray(mat, dtype="float32", order="C")
    dim = mat.shape[1]

    with open(path, "wb") as f:
        hdr = np.int32(dim).tobytes()
        for row in mat:
            f.write(hdr)
            f.write(row.tobytes())


def embed_and_export_fvecs(
    corpus_sentences: list[str],
    questions: list[str],
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
    out_prefix: str = "squad",
    batch_size: int = 64,
):
    """
    • corpus_sentences → <prefix>_base.fvecs
    • questions        → <prefix>_query.fvecs
    Both files follow the .fvecs spec (little‑endian).
    """
    model = SentenceTransformer(model_name)

    print(f"🔢  Encoding {len(corpus_sentences):,d} corpus sentences …")
    base_vecs = model.encode(
        corpus_sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"❓  Encoding {len(questions):,d} questions …")
    query_vecs = model.encode(
        questions,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    base_path  = f"{out_prefix}_base.fvecs"
    query_path = f"{out_prefix}_query.fvecs"
    print("💾  Writing", base_path, "and", query_path)
    write_fvecs(base_path,  base_vecs)
    write_fvecs(query_path, query_vecs)
    print("🏁  Done – files are ready for Go HNSW / Faiss etc.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed SQuAD into .fvecs")
    parser.add_argument("squad_json", help="Path to SQuAD v1/v2 JSON file")
    parser.add_argument("--model",    default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Sentence‑Transformers model name")
    parser.add_argument("--prefix",   default="squad_emb",
                        help="Output path prefix (default: squad_emb)")
    args = parser.parse_args()

    squad_to_fvecs(args.squad_json, args.model, args.prefix)
