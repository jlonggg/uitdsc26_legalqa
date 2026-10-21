"""Đoạn trích (quote) của nền v11:
    · widen: bắt đầu tại chunk thắng trong Điều của nó, dừng ở ranh giới Điều, tối đa TARGET_WORDS=550 từ.
    · k4_trim: khi Điều dài hơn đoạn trích quá 80 từ và đoạn trích có ≥ 3 khoản, cắt đuôi sau khoản cuối cùng còn được
      một chunk cùng văn bản trong pool hỗ trợ với điểm ≥ K4_FRAC=0,85 × điểm cao nhất (chỉ cắt đuôi, giữ tối thiểu 250 từ)."""
from __future__ import annotations

import re

from config import (K4_FRAC, K4_MIN_KEEP, K4_PAD, MIN_WORDS, TARGET_WORDS)

_KHOAN = re.compile(r"^\d{1,2}\.$")


def _find_sub(hay: list[str], needle: list[str]) -> int:
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return -1
    f = needle[0]
    for i in range(n - m + 1):
        if hay[i] == f and hay[i:i + m] == needle:
            return i
    return -1


def khoan_units(toks: list[str]) -> list[tuple[int, int]]:
    marks = [i for i, t in enumerate(toks) if _KHOAN.match(t)]
    if not marks:
        return [(0, len(toks))]
    bs = [0] + marks + [len(toks)]
    return [(bs[i], bs[i + 1]) for i in range(len(bs) - 1) if bs[i + 1] > bs[i]]


def widen(passage: str, span, chunk_text: str,
          n_words: int = TARGET_WORDS) -> str:
    cs, ce = span
    if cs is None or cs < 0 or cs >= len(passage):
        return " ".join(chunk_text.split()[:n_words])
    end = ce if (ce and cs < ce <= len(passage)) else len(passage)
    dieu = passage[cs:end].split()
    if not dieu:
        return " ".join(chunk_text.split()[:n_words])
    ct = chunk_text.split()
    pos = _find_sub(dieu, ct[:8]) if len(ct) >= 8 else 0
    start = max(0, pos) if pos >= 0 else 0
    return " ".join(dieu[start:start + max(MIN_WORDS, n_words)])


def k4_trim(quote: str, dieu_words: list[str],
            doc_chunks: list[tuple[str, float]]) -> str:
    qt = quote.split()
    if not doc_chunks or len(dieu_words) <= len(qt) + K4_PAD:
        return quote
    wpos = _find_sub(dieu_words, qt[:8]) if len(qt) >= 8 else -1
    if wpos < 0:
        return quote
    spans = []
    for ct, sc in doc_chunks:
        cw = str(ct).split()
        p = _find_sub(dieu_words, cw[:8]) if len(cw) >= 8 else -1
        if p >= 0:
            spans.append((p, p + len(cw), float(sc)))
    if not spans:
        return quote
    units = khoan_units(qt)
    if len(units) < 3:
        return quote
    mx = max(s for _, _, s in spans)
    if mx <= 0:
        return quote
    sup = []
    for s, e in units:
        a, b = wpos + s, wpos + e
        sup.append(max((sc for cs2, ce2, sc in spans
                        if cs2 < b and ce2 > a), default=0.0) / mx)
    last = max((j for j, s in enumerate(sup) if s >= K4_FRAC), default=None)
    if last is None or last >= len(units) - 1:
        return quote
    keep_end = units[last][1]
    if keep_end < K4_MIN_KEEP:
        return quote
    return " ".join(qt[:keep_end])
