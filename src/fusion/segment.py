"""Chunk store của nền v11: tách mỗi passage theo mốc "Điều N." (phần mở đầu > 200 ký tự thành 1 segment riêng);
Điều dài hơn MAX_CHUNK_WORDS=250 từ được cắt thành chunk con cửa sổ 250 từ, bước SPLIT_STRIDE=200. Mọi chunk con mang nhãn
"Điều N" và span ký tự của Điều mẹ để widen() quay lại được nguyên văn Điều.
-> WORK/fusion/index/chunks.pkl"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field

from config import ART, MAX_CHUNK_WORDS, SPLIT_STRIDE
from corpus import load_corpus, load_passage

_DIEU = re.compile(r"(?m)^\s*(Điều\s+\d+[a-zđ]?)\s*[\.:]")


@dataclass
class ChunkStore:
    doc_ids: list[str] = field(default_factory=list)     # theo doc index
    titles: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)       # theo chunk index
    labels: list[str] = field(default_factory=list)      # "Điều N" của mẹ
    spans: list[tuple] = field(default_factory=list)     # (cs, ce) char Điều mẹ
    doc_of_chunk: list[int] = field(default_factory=list)


def segment_doc(psg: str) -> list[tuple[str, int, int]]:
    """-> [(label, cs, ce)] các Điều; không mốc nào -> cả passage một segment."""
    ms = list(_DIEU.finditer(psg))
    if not ms:
        return [("", 0, len(psg))]
    segs = []
    for i, m in enumerate(ms):
        cs = m.start()
        ce = ms[i + 1].start() if i + 1 < len(ms) else len(psg)
        segs.append((m.group(1), cs, ce))
    if ms[0].start() > 200:                     # phần mở đầu trước Điều 1
        segs.insert(0, ("", 0, ms[0].start()))
    return segs


def build(store_path=None) -> ChunkStore:
    corpus = load_corpus()
    st = ChunkStore()
    for di, (doc_id, meta) in enumerate(sorted(corpus.items())):
        st.doc_ids.append(doc_id)
        st.titles.append(meta["title"])
        psg = load_passage(meta["path"])
        for label, cs, ce in segment_doc(psg):
            words = psg[cs:ce].split()
            if len(words) < 8:
                continue
            if len(words) <= MAX_CHUNK_WORDS:
                parts = [words]
            else:
                parts = [words[s:s + MAX_CHUNK_WORDS]
                         for s in range(0, len(words), SPLIT_STRIDE)
                         if len(words[s:s + MAX_CHUNK_WORDS]) >= 40]
            for pw in parts:
                st.texts.append(" ".join(pw))
                st.labels.append(label)
                st.spans.append((cs, ce))       # span ĐIỀU MẸ cho mọi chunk con
                st.doc_of_chunk.append(di)
    if store_path:
        with open(store_path, "wb") as fh:
            pickle.dump({"doc_ids": st.doc_ids, "titles": st.titles,
                         "texts": st.texts, "labels": st.labels,
                         "spans": st.spans,
                         "doc_of_chunk": st.doc_of_chunk}, fh, protocol=4)
    return st


def load_store() -> ChunkStore:
    """Loader dùng chung — pickle là dict thuần, không ghim module."""
    with open(ART / "chunks.pkl", "rb") as fh:
        d = pickle.load(fh)
    return ChunkStore(**d)


if __name__ == "__main__":
    ART.mkdir(parents=True, exist_ok=True)
    st = build(ART / "chunks.pkl")
    print(f"{len(st.doc_ids)} văn bản -> {len(st.texts)} chunk "
          f"(≈{len(st.texts)/len(st.doc_ids):.1f} chunk/vb)")
