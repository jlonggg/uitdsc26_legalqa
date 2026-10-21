#!/usr/bin/env python3
"""Bước R0 (CPU): dựng 3 bảng dẫn xuất corpus từ `data/selected-contexts`.

  corpus.jsonl   = catalog baseline (8.512 văn bản có passage, 347.396 chunk; baseline_common.chunk_document 1500/1800 ký tự)
                   -> nguồn của ES `dsc26_legal_chunks_v1` và `dsc26_legal_chunks_wseg_h77` (index_es.py)
                   Kiểm 29/09/2026: sha256 `a5a7d8d0…` trùng catalog đã dùng cho bài nộp.
  chunks.jsonl   = (id, doc_id, article, order, name, text) của mọi chunk (cùng nội dung index ES dsc26_legal_chunks_v1).
  articles.jsonl = 192.670 đơn vị cấp Điều của `legal_units.parse_legal_units` (đầu vào của Qwen3-Embedding, dense_qwen3.py)
                   Kiểm 29/09/2026: 192.670/192.670 đơn vị trùng unit_id + mọi trường với bản đã dùng cho bài nộp.

usage: build_catalog.py <out_dir>
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import build_chunks, load_contexts, readable_title, write_corpus_catalog   # noqa: E402
from legal_units import parse_legal_units                                                      # noqa: E402

DATA = Path(__import__('os').environ.get('DATA', str(Path(__file__).resolve().parents[2] / 'data')))


def doc_title(d):
    """Tiêu đề văn bản của đơn vị cấp Điều: `name` (slug) đọc được; không có `name` thì lấy slug của link (bỏ số trang cuối)."""
    if d.name:
        return readable_title(d.name)
    tail = d.link.rstrip('/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
    return re.sub(r'[-_]+', ' ', re.sub(r'-\d+$', '', tail)).strip()


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    docs = load_contexts(DATA / 'selected-contexts')
    chunks = build_chunks(docs, {'chunking': {'target_chars': 1500, 'max_chars': 1800}})
    write_corpus_catalog(docs, chunks, {'base_dir': str(out), 'paths': {'artifacts': '.'}})
    with open(out / 'chunks.jsonl', 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps({'id': c.chunk_id, 'doc_id': c.doc_id, 'article': c.article, 'order': c.order,
                                'name': c.name, 'text': c.text}, ensure_ascii=False) + '\n')
    n = 0
    with open(out / 'articles.jsonl', 'w', encoding='utf-8') as f:
        for d in docs:
            for u in parse_legal_units(d.doc_id, doc_title(d), d.passage):
                if u.level != 'article':
                    continue
                f.write(json.dumps({'doc_id': u.doc_id, 'unit_id': u.unit_id, 'inst': u.structure_instance, 'article': u.article,
                                    'header': u.parent_header, 'body': u.body, 'boilerplate': u.boilerplate or ''},
                                   ensure_ascii=False) + '\n')
                n += 1
    print('docs', len(docs), 'chunks', len(chunks), 'articles', n)
    assert (len(docs), len(chunks), n) == (8512, 347396, 192670), 'corpus khác bản BTC đã dùng'


if __name__ == '__main__':
    main(sys.argv[1])
