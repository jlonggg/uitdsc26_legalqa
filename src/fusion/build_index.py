"""Chỉ mục nền v11 trên chunk store (segment.py): BM25 (bm25s, k1=1,2, b=1,0) và dense Vietnamese_Embedding fp16 (chuẩn hoá L2).
Text chỉ mục = tiêu đề văn bản + nội dung chunk.

Chạy:  python3 build_index.py --bm25     -> WORK/fusion/index/bm25s_index/
       python3 build_index.py --dense    -> WORK/fusion/index/emb_aiteam.npy (GPU, ~1 giờ trên RTX 2080 Ti)"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from config import ART, BM25_B, DENSE_MODEL
from segment import load_store


def index_texts(st) -> list[str]:
    return [f"{st.titles[st.doc_of_chunk[ci]]} {st.texts[ci]}"
            for ci in range(len(st.texts))]


def tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", action="store_true")
    ap.add_argument("--dense", action="store_true")
    args = ap.parse_args()
    st = load_store()
    texts = index_texts(st)
    print(f"{len(texts)} chunk", flush=True)

    if args.bm25:
        import bm25s
        retr = bm25s.BM25(k1=1.2, b=BM25_B)
        retr.index(bm25s.tokenize(texts, stopwords=None, show_progress=False))
        retr.save(str(ART / "bm25s_index"))
        print("đã ghi bm25s_index/", flush=True)

    if args.dense:
        import torch
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(DENSE_MODEL, device="cuda")
        enc.half()
        out = np.lib.format.open_memmap(
            ART / "emb_aiteam.tmp.npy", mode="w+", dtype=np.float16,
            shape=(len(texts), 1024))
        B = 512
        import time
        t0 = time.time()
        for s in range(0, len(texts), B):
            V = enc.encode(texts[s:s + B], batch_size=64,
                           normalize_embeddings=True, show_progress_bar=False)
            out[s:s + len(V)] = V.astype(np.float16)
            if s and s % (B * 20) == 0:
                r = (time.time() - t0) / s
                print(f"  {s}/{len(texts)} ({r*1000:.1f} ms/chunk, còn "
                      f"{(len(texts)-s)*r/60:.0f} phút)", flush=True)
        out.flush(); del out
        (ART / "emb_aiteam.tmp.npy").replace(ART / "emb_aiteam.npy")   # chỉ xuất hiện khi encode xong (resume an toàn)
        print("đã ghi emb_aiteam.npy", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
