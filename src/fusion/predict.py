"""Nền v11, bước GPU: với mỗi câu hỏi test
    retrieve (BM25 + dense -> cross-encoder -> chọn Điều) -> evidence (widen + K4) -> compose (khung đáp án)
    -> memory replay (câu gần trùng câu TRAIN, xem memory.py) được ưu tiên trước retrieval.

Chạy:  python3 predict.py --out WORK/fusion/pred_shard0.json --shard 0/4 [--margins-out ...] [--no-memory] [--limit N]"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TEST_JSON
from compose import compose
from corpus import load_corpus, load_passage
from evidence import k4_trim, widen
from memory import Memory
from retrieve import Retriever


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--margins-out", default=None,
                    help="ghi margin top1-top2 theo qid (chỉ để chẩn đoán; bản v20 không dùng)")
    ap.add_argument("--shard", default="",
                    help="i/n: chỉ chạy shard i trong n (round-robin trên qid đã sort)")
    args = ap.parse_args()

    test = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    qids = sorted(test)
    if args.limit:
        qids = qids[: args.limit]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        qids = qids[i::n]
    corpus = load_corpus()
    r = Retriever(use_ce=os.environ.get("USE_CE", "1") == "1")   # USE_CE=0: ablation bỏ cross-encoder
    mem = None if args.no_memory else Memory(encoder=r.enc)

    out, margins = {}, {}
    n_mem = 0
    t0 = time.time()
    cache: dict[str, str] = {}
    for k, qid in enumerate(qids):
        q = str(test[qid]["question"])
        hit = mem.lookup(q) if mem else None
        if hit:
            out[qid] = {"answer": hit[0]}
            n_mem += 1
            continue
        di, ci, pool, doc_chunks, margin = r.pick(q)
        margins[qid] = margin
        doc_id = r.st.doc_ids[di]
        meta = corpus.get(doc_id, {})
        if doc_id not in cache:
            cache.clear()
            cache[doc_id] = load_passage(meta["path"]) if meta.get("path") \
                else ""
        psg = cache[doc_id]
        span = r.st.spans[ci]
        quote = widen(psg, span, r.st.texts[ci])
        cs, ce = span
        if cs is not None and 0 <= cs < len(psg):
            end = ce if (ce and cs < ce <= len(psg)) else len(psg)
            quote = k4_trim(quote, psg[cs:end].split(), doc_chunks)
        cite = r.st.labels[ci] or ""
        docnum = (meta.get("docnums") or [""])[0]
        out[qid] = {"answer": compose(q, cite, docnum,
                                      meta.get("title", ""), quote)}
        if k and k % 100 == 0:
            print(f"  {k}/{len(qids)} ({(time.time()-t0)/k:.2f}s/câu)",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False),
                              encoding="utf-8")
    print(f"đã ghi {args.out} · memory-replay {n_mem} câu", flush=True)
    if args.margins_out:
        Path(args.margins_out).write_text(json.dumps(margins),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
