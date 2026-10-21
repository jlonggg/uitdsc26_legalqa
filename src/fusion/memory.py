"""Memory replay: nếu câu hỏi test trùng (sau chuẩn hoá) hoặc có cosine Vietnamese_Embedding ≥ 0,95 với một câu hỏi TRAIN
và hai câu có cùng thực thể (các con số, cấp hành chính) thì dùng lại nguyên văn đáp án TRAIN đó. Chỉ đọc train_QA.json."""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from config import ADMIN_TERMS, DENSE_MODEL, MEM_THRESHOLD, TRAIN_JSON


def norm_q(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[?.!,;:\"'()\[\]]", " ", s)
    return " ".join(s.split())


def _ents(q: str):
    ql = norm_q(q)
    nums = frozenset(re.findall(r"\d+(?:/\d+)?(?:[/-][\w.]+)*", ql))
    adm = frozenset(a for a in ADMIN_TERMS if a in ql)
    return nums, adm


class Memory:
    def __init__(self, encoder=None, device: str = "cuda"):
        import json
        self.t2 = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
        self.ids = sorted(self.t2)
        self.norm = {}
        for qid in self.ids:
            self.norm.setdefault(norm_q(self.t2[qid]["question"]), qid)
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(DENSE_MODEL, device=device)
            if device == "cuda":
                encoder.half()
        self.enc = encoder
        self.tv = self.enc.encode(
            [str(self.t2[k]["question"]) for k in self.ids],
            batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def lookup(self, question: str):
        """-> (answer, train_qid, sim) nếu replay được, else None."""
        hit = self.norm.get(norm_q(question))
        sim = 1.0
        if hit is None:
            qv = self.enc.encode([question], normalize_embeddings=True)[0]
            sims = self.tv @ qv
            j = int(np.argmax(sims))
            sim = float(sims[j])
            if sim < MEM_THRESHOLD:
                return None
            hit = self.ids[j]
        mem = str(self.t2[hit].get("answer") or "").strip()
        if not mem:
            return None
        n1, a1 = _ents(question)
        n2, a2 = _ents(str(self.t2[hit]["question"]))
        if n1 != n2 or a1 != a2:                     # chốt xung đột thực thể
            return None
        return mem, hit, sim
