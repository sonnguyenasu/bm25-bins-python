import json, pathlib, argparse, re
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import load_dataset


def _write_fvecs(path: str | pathlib.Path, mat: np.ndarray) -> None:
    mat = np.asarray(mat, dtype="float32", order="C")
    dim = np.int32(mat.shape[1]).tobytes()
    with open(path, "wb") as f:
        for row in mat:
            f.write(dim)
            f.write(row.tobytes())


def _write_ivecs(path: str | pathlib.Path, mat: np.ndarray) -> None:
    mat = np.asarray(mat, dtype="int32", order="C")
    dim = np.int32(mat.shape[1]).tobytes()
    with open(path, "wb") as f:
        for row in mat:
            f.write(dim)
            f.write(row.tobytes())



def build_corpus_and_groundtruth(split, k):
    """
    • Returns  corpus_sentences (list[str])
             ,  questions        (list[str])
             ,  gnd_arr          (nQuery, k) int32   (‑1 padded)
    """
    stem_sentence = lambda s: re.sub(r"\s+", " ", s.strip())

    corpus_set       = set()
    question_texts   = []
    gnd_lists        = []                 # list[list[int]]
    sent_index_cache = {}                 # sentence -> idx in corpus_list

    print("🔄  Building sentence corpus and ground‑truth map …")
    for ex in tqdm(split):
        q            = ex["question"].strip()
        answers = [a.strip() for a in ex["answers"]["text"]]
        context_sent = [stem_sentence(s) for s in ex["context"].split(".") if s.strip()]

        # add sentences to corpus, remember their indices
        indices = []
        for sent in context_sent:
            if sent not in sent_index_cache:
                sent_index_cache[sent] = len(corpus_set)
                corpus_set.add(sent)
            idx = sent_index_cache[sent]
            # if any gold answer substring occurs in this sentence, mark for GT
            if any(a in sent for a in answers):
                indices.append(idx)

        # keep at most k GT neighbours, pad with ‑1
        if len(indices) == 0:
            indices = [-1]
        indices = indices[:k] + [-1]*(k - len(indices))
        gnd_lists.append(indices)

        question_texts.append(q)

    corpus_list = list(corpus_set)
    gnd_arr     = np.array(gnd_lists, dtype="int32")
    return corpus_list, question_texts, gnd_arr


def export_squad_embeddings(
    split_name: str       = "validation",
    k: int                = 100,
    model_name: str       = "mixedbread-ai/mxbai-embed-large-v1",
    out_prefix: str       = "squad_val",
    batch_size: int       = 64,
):
    ds        = load_dataset("rajpurkar/squad_v2", split=split_name)
    corpus, questions, gnd = build_corpus_and_groundtruth(ds, k)

    model = SentenceTransformer(model_name)

    print(f"🔢  Encoding {len(corpus):,d} corpus sentences …")
    base_vecs = model.encode(
        corpus, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )

    print(f"❓  Encoding {len(questions):,d} questions …")
    query_vecs = model.encode(
        questions, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )

    _write_fvecs(f"{out_prefix}_base.fvecs",  base_vecs)
    _write_fvecs(f"{out_prefix}_query.fvecs", query_vecs)
    _write_ivecs(f"{out_prefix}_gnd.ivecs",   gnd)

    print("🏁  Wrote")
    print("   ", f"{out_prefix}_base.fvecs  ({base_vecs.shape[0]} × {base_vecs.shape[1]})")
    print("   ", f"{out_prefix}_query.fvecs ({query_vecs.shape[0]} × {query_vecs.shape[1]})")
    print("   ", f"{out_prefix}_gnd.ivecs   ({gnd.shape[0]}    × {gnd.shape[1]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embed SQuAD + produce ground‑truth .ivecs")
    ap.add_argument("--split", default="validation",
                    help="SQuAD split (train/validation)")
    ap.add_argument("--k",     type=int, default=5,
                    help="How many GT neighbours per query (matches Go ‑k)")
    ap.add_argument("--model", default="mixedbread-ai/mxbai-embed-large-v1",
                    help="Sentence‑Transformers model")
    ap.add_argument("--prefix", default="squad_val",
                    help="Output filename prefix")
    args = ap.parse_args()

    export_squad_embeddings(
        split_name=args.split,
        k=args.k,
        model_name=args.model,
        out_prefix=args.prefix,
    )
