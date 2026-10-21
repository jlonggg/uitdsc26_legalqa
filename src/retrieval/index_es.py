#!/usr/bin/env python3
"""Stage R1 (CPU): nạp 2 index Elasticsearch 7.17 mà pool_wseg.py / score_ce.py truy vấn, từ corpus.jsonl (build_catalog.py).

  dsc26_legal_chunks_v1        chunk nguyên văn (name, article, order, text) — score_ce.py lấy text ứng viên từ đây
  dsc26_legal_chunks_wseg_h77  văn bản đã tách từ underthesea của chuỗi "name article [title] text" — pool_wseg.py truy vấn BM25 ở đây
     Index dùng cho bài nộp được dựng 2 đợt: FIRST_PASS=314.000 chunk ĐẦU (thứ tự corpus.jsonl) KHÔNG kèm `title`,
     phần còn lại CÓ kèm `title`. Giữ nguyên luật này để tái lập đúng index (kiểm 29/09/2026: 400/400 chunk mẫu khớp).

usage: index_es.py <corpus.jsonl> [--workers N]      (ES_URL mặc định http://127.0.0.1:9211; index đã đủ số chunk thì bỏ qua)
"""
import argparse, json, os, signal, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import lexical_text, title_text   # noqa: E402
from elasticsearch import Elasticsearch, helpers

ES_URL = os.environ.get('ES_URL', 'http://127.0.0.1:9211')
LEGACY, WSEG, N_CHUNKS, FIRST_PASS = 'dsc26_legal_chunks_v1', 'dsc26_legal_chunks_wseg_h77', 347396, 314000


class SegmentTimeout(Exception): pass
def _timeout(*_): raise SegmentTimeout()


def segment(text, timeout=8):
    """underthesea word_tokenize; quá 8 s thì giữ nguyên chuỗi (gộp khoảng trắng)."""
    from underthesea import word_tokenize
    old = signal.signal(signal.SIGALRM, _timeout); signal.alarm(timeout)
    try: return word_tokenize(text, format='text')
    except SegmentTimeout: return ' '.join(text.split())
    finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)


def wseg_row(job):
    i, c = job
    parts = [c['name'], c['article']] + ([title_text(c['name'])] if i >= FIRST_PASS else []) + [c['text']]
    return c['chunk_id'], c['doc_id'], segment(' '.join(filter(None, parts)))


def chunks(path):
    with open(path, encoding='utf-8') as f:
        f.readline()                                   # dòng meta (documents)
        for line in f:
            yield json.loads(line)


def full(es, index):
    return es.indices.exists(index=index) and es.count(index=index)['count'] == N_CHUNKS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('corpus'); ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    es = Elasticsearch(ES_URL, request_timeout=180, retry_on_timeout=True, max_retries=5)
    if not full(es, LEGACY):
        if es.indices.exists(index=LEGACY): es.indices.delete(index=LEGACY)
        es.indices.create(index=LEGACY, body={'settings': {'refresh_interval': '-1'}, 'mappings': {'dynamic': 'strict', 'properties': {
            'doc_id': {'type': 'keyword'}, 'name': {'type': 'text'}, 'article': {'type': 'keyword'}, 'order': {'type': 'integer'},
            'lexical': {'type': 'text', 'analyzer': 'whitespace'}, 'title': {'type': 'text', 'analyzer': 'whitespace'},
            'text': {'type': 'text', 'index': False}}}})
        actions = ({'_index': LEGACY, '_id': c['chunk_id'], '_source': {'doc_id': c['doc_id'], 'name': c['name'], 'article': c['article'],
                    'order': c['order'], 'lexical': lexical_text(f"{c['name']} {c['article']} {c['text']}"), 'title': title_text(c['name']),
                    'text': c['text']}} for c in chunks(a.corpus))
        for ok, _ in helpers.streaming_bulk(es, actions, chunk_size=500, request_timeout=180): pass
        es.indices.put_settings(index=LEGACY, body={'index': {'refresh_interval': '1s'}}); es.indices.refresh(index=LEGACY)
    print(LEGACY, es.count(index=LEGACY)['count'], flush=True)
    if not full(es, WSEG):
        if es.indices.exists(index=WSEG): es.indices.delete(index=WSEG)
        es.indices.create(index=WSEG, body={
            'settings': {'number_of_shards': 1, 'number_of_replicas': 0, 'refresh_interval': '-1'},
            'mappings': {'properties': {'doc_id': {'type': 'keyword'}, 'wseg': {'type': 'text'}}}})
        t0 = time.time()
        with Pool(a.workers) as pool:
            rows = pool.imap(wseg_row, enumerate(chunks(a.corpus)), chunksize=64)   # imap giữ thứ tự corpus
            actions = ({'_index': WSEG, '_id': cid, '_source': {'doc_id': doc, 'wseg': v}} for cid, doc, v in rows)
            for k, _ in enumerate(helpers.streaming_bulk(es, actions, chunk_size=256, request_timeout=180), 1):
                if k % 20000 == 0: print(f'wseg {k}/{N_CHUNKS} {k / (time.time() - t0):.0f}/s', flush=True)
        es.indices.put_settings(index=WSEG, body={'index': {'refresh_interval': '1s'}}); es.indices.refresh(index=WSEG)
    print(WSEG, es.count(index=WSEG)['count'], flush=True)
    assert full(es, LEGACY) and full(es, WSEG)


if __name__ == '__main__':
    main()
