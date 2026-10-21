"""Đọc 8.532 file context_*.json của BTC (selected-contexts): tiêu đề văn bản lấy từ khối chữ IN HOA có dấu ở ~400 ký tự
đầu passage, không có thì dựng từ slug của link; số hiệu văn bản (vd "21/2021/NĐ-CP") giữ nguyên văn, tối đa 3 số hiệu
trong 1.500 ký tự đầu."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from config import CORPUS_DIR

_DOCNUM = re.compile(r"\b(\d{1,4}/\d{4}/[A-ZĐ][\w\-ĐđĂăÂâÊêÔôƠơƯư]*(?:-[A-ZĐ][\w]*)*)\b")
_VIET = set("ăâêôơưđáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợ"
            "úùủũụứừửữựýỳỷỹỵ")


def _has_viet(s: str) -> bool:
    return any(c in _VIET for c in s.lower())


def _title_from_passage(psg: str) -> str | None:
    """Khối VIẾT HOA CÓ DẤU trong ~400 ký tự đầu = tiêu đề thật."""
    head = " ".join(psg[:400].split())
    toks = head.split()
    run: list[str] = []
    best: list[str] = []
    for t in toks[:60]:
        core = re.sub(r"[^\wĐđ]", "", t)
        if core and (core.isupper() or core.isdigit() or "/" in t):
            run.append(t)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    # bỏ token số/năm/số-hiệu dẫn đầu — title phải mở bằng LOẠI văn bản
    while best and (best[0].replace(".", "").isdigit() or "/" in best[0]):
        best = best[1:]
    if len(best) >= 3 and _has_viet(" ".join(best)):
        return " ".join(best)[:180]
    return None


def _title_from_link(link: str) -> str:
    slug = link.rstrip("/").split("/")[-1]
    slug = re.sub(r"\.(html?|aspx)$", "", slug)
    return " ".join(slug.split("-"))[:180]


def load_corpus() -> dict[str, dict]:
    """-> {doc_id: {title, title_src, docnums, path}} — passage đọc lười."""
    out: dict[str, dict] = {}
    for p in sorted(CORPUS_DIR.glob("context_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        doc_id = str(d.get("id") or p.stem.split("_", 1)[-1])
        psg = unicodedata.normalize("NFC", str(d.get("passage") or ""))
        t_psg = _title_from_passage(psg)
        title = t_psg or _title_from_link(str(d.get("link") or ""))
        dns = list(dict.fromkeys(_DOCNUM.findall(psg[:1500])))
        out[doc_id] = {"title": title,
                       "title_src": "passage" if t_psg else "slug",
                       "docnums": dns[:3], "path": str(p)}
    return out


def load_passage(path: str) -> str:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return unicodedata.normalize("NFC", str(d.get("passage") or ""))


if __name__ == "__main__":
    c = load_corpus()
    n = len(c)
    n_psg = sum(1 for v in c.values() if v["title_src"] == "passage")
    n_dn = sum(1 for v in c.values() if v["docnums"])
    print(f"{n} văn bản · title-có-dấu-từ-passage {n_psg} ({100*n_psg/n:.1f}%)"
          f" · có docnum {n_dn} ({100*n_dn/n:.1f}%)")
    for k in list(c)[:3]:
        print(f"  {k}: [{c[k]['title_src']}] {c[k]['title'][:70]} "
              f"| {c[k]['docnums']}")
