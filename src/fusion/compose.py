"""Dựng khung đáp án từ đoạn trích nguyên văn:
    dòng dẫn "Căn cứ theo quy định tại khoản N Điều M <số hiệu> <tên văn bản>" -> câu mốc (MARKER)
    -> các gạch đầu dòng diễn giải lấy từ đoạn trích -> đoạn trích nguyên văn -> câu kết "Như vậy, <câu hỏi>" -> dòng đuôi.
Kèm: cắt câu cụt ở cuối đoạn trích/gạch đầu dòng, bỏ lặp "Điều N" ở đầu đoạn trích nếu dòng dẫn đã nêu."""
from __future__ import annotations

import re

from config import (BOILER_TAIL, MARKER, RESTATE_WORDS_PER_PART, TARGET_WORDS,
                    TRIM_DANGLE, TRIM_KEEP_QUOTE, TRIM_KEEP_TAIL)

_VIET = set("ăâêôơưđáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợ"
            "úùủũụứừửữựýỳỷỹỵ")
_DOCTYPES = ["thông tư liên tịch", "bộ luật", "thông tư", "nghị định",
             "nghị quyết", "quyết định", "chỉ thị", "pháp lệnh", "hiến pháp",
             "công văn", "luật"]
_QTAIL = re.compile(
    r"\s*(?:là\s+(?:bao\s+lâu|bao\s+nhiêu|gì|ai|khi\s+nào|ở\s+đâu|"
    r"như\s+thế\s+nào|thế\s+nào)"
    r"|có\s+(?:đúng\s+)?(?:không|phải\s+không)|hay\s+không|được\s+không"
    r"|phải\s+không|đúng\s+không|như\s+thế\s+nào|thế\s+nào|ra\s+sao"
    r"|bao\s+lâu|bao\s+nhiêu|khi\s+nào|ở\s+đâu|không)\s*[?.!]*\s*$", re.I)
_BULLET = re.compile(r"(?m)^\s*(\d+[\.\)]|[a-zđ][\.\)]|-)\s+")
_DIEU_PRE = re.compile(r"^Điều\s+\d+[a-zđ]?\s*[\.:]?\s*")
_KHOAN_N = re.compile(r"(?<![/\d])(\d{1,2})[\.\)]\s")
_SENT_END = re.compile(r"[\.;:]$")


def _has_viet(s: str) -> bool:
    return any(c in _VIET for c in s.lower())


def _type_prefix(toks: list[str]) -> int:
    low = " ".join(toks).lower()
    for t in _DOCTYPES:
        if low.startswith(t):
            return len(t.split())
    return 0


def khoan_num(quote: str) -> str:
    m = _KHOAN_N.search(_DIEU_PRE.sub("", quote[:400]))
    return m.group(1) if m else "1"


def build_header(cite: str, docnum: str, title: str, quote: str) -> str:
    """Dòng dẫn "Căn cứ theo quy định tại khoản N <Điều M> <loại văn bản> <số hiệu> <phần còn lại của tiêu đề>".
    Tiêu đề không dấu (dựng từ slug link) thì bỏ. Có số hiệu thì không thêm 'quy định như sau' vì câu mốc
    MARKER ngay sau đã nói; không có số hiệu thì dùng "... <tiêu đề> quy định như sau:"."""
    title_s = " ".join((title or "").split())
    if docnum:
        tt = title_s.split()
        if tt and not _has_viet(title_s):
            tt = []                                  # tiêu đề không dấu = slug, bỏ
        k = _type_prefix(tt)
        body = (tt[:k] + [docnum] + tt[k:]) if k else [docnum] + tt
        head = " ".join([f"Căn cứ theo quy định tại khoản {khoan_num(quote)}"]
                        + ([cite] if cite else []) + body)
    else:
        head = ("Căn cứ " + cite + " " if cite else "Căn cứ quy định tại ")
        head += f"{title_s} quy định như sau:"
        if head.startswith("Căn cứ ") \
                and not head.startswith("Căn cứ quy định tại"):
            head = (f"Căn cứ theo quy định tại khoản {khoan_num(quote)}"
                    + head[len("Căn cứ"):])
    return " ".join(head.split())


def restate(quote: str, n_words: int) -> str:
    parts = [p.strip() for p in _BULLET.split(quote)
             if p and len(p.strip().split()) >= 6]
    if len(parts) < 2:
        parts = [p.strip() for p in re.split(r"(?<=[\.;])\s+", quote)
                 if len(p.strip().split()) >= 6]
    out, used = [], 0
    for p in parts:
        w = p.split()
        if not w:
            continue
        out.append("- " + " ".join(w[:RESTATE_WORDS_PER_PART]))
        used += min(len(w), RESTATE_WORDS_PER_PART)
        if used >= n_words:
            break
    return "\n".join(out)


def trim_sentence(text: str, max_dangle: int, min_keep: int) -> str:
    w = text.split()
    for i in range(len(w) - 1, -1, -1):
        if _SENT_END.search(w[i]):
            d = len(w) - 1 - i
            if 0 < d <= max_dangle and i + 1 >= min_keep:
                return " ".join(w[:i + 1])
            return text
    return text


def echo(question: str) -> str:
    q = question.strip()
    if q.rstrip("? ").lower().endswith(("không", "phải không", "đúng không")):
        c = _QTAIL.sub("", q.rstrip("? ")).strip().rstrip("?,.;: ")
        return f"Như vậy, {c}" if c else f'Như vậy, {q.rstrip("? ")}'
    return f'Như vậy, {q.rstrip("? ")}'


def compose(question: str, cite: str, docnum: str, title: str,
            quote: str, target: int = TARGET_WORDS) -> str:
    head = build_header(cite, docnum, title, quote)
    tail = restate(quote, max(60, target - len(quote.split())))
    # header đã nêu "Điều N" -> bỏ lặp ở đầu đoạn trích
    m = re.match(r"^(Điều\s+\d+[a-zđ]?)", quote)
    if m and m.group(1) in head:
        quote = re.sub(rf"^{re.escape(m.group(1))}\s*[.:]?\s*", "", quote)
    # cắt câu cụt (SAU khi đã dựng gạch đầu dòng từ đoạn trích chưa cắt)
    quote = trim_sentence(quote, TRIM_DANGLE, TRIM_KEEP_QUOTE)
    tail = "\n".join(trim_sentence(l, TRIM_DANGLE, TRIM_KEEP_TAIL)
                     for l in tail.split("\n"))
    ket = echo(question)
    body = f"{head}\n{MARKER}\n{tail}\n{quote}"
    return f"{body}\n{ket}\n{BOILER_TAIL}"
