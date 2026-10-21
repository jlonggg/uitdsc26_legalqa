#!/usr/bin/env python3
"""Fuse wseg-BM25 doc pool with Qwen3 dense (MaxP) via RRF; write a pool in the same format as pool_wseg.py (input of score_ce.py); no labels used.
Mỗi văn bản giữ tối đa 3 chunk: chunk đầu của các Điều dense top-60 trước, rồi 3 chunk BM25 (inner_hits).
usage: fuse.py <pool.jsonl> <dense.jsonl> <out_pool.jsonl> <chunks.jsonl> [k=60]"""
import json, re, sys, collections
pool_p, dense_p, out_p, chunks_p = sys.argv[1:5]; K = int(sys.argv[5]) if len(sys.argv) > 5 else 60
first_chunk = {}
for l in open(chunks_p, encoding='utf-8'):
    r = json.loads(l); m = re.match(r'Điều\s+(\d+[a-zđ]?)', r['article'] or '')
    if m:
        key = (r['doc_id'], m.group(1).lower())
        if key not in first_chunk or r['order'] < first_chunk[key][1]: first_chunk[key] = (r['id'], r['order'])
dense = {json.loads(l)['qid']: json.loads(l) for l in open(dense_p, encoding='utf-8')}
with open(out_p, 'w', encoding='utf-8') as out:
    for l in open(pool_p, encoding='utf-8'):
        row = json.loads(l); qid = row['qid']; d = dense.get(qid)
        if not d: continue
        bm = [x['doc_id'] for x in row['docs']]; dn = [x[0] for x in d['docs']]
        sc = collections.defaultdict(float)
        for i, x in enumerate(bm): sc[x] += 1 / (K + i + 1)
        for i, x in enumerate(dn): sc[x] += 1 / (K + i + 1)
        fused = sorted(sc, key=lambda x: -sc[x])[:100]
        bm_chunks = {x['doc_id']: x for x in row['docs']}
        dense_art = collections.defaultdict(list)
        for doc, art, uid, s in d['arts']:
            m = re.match(r'Điều\s+(\d+[a-zđ]?)', art)
            if m and (doc, m.group(1).lower()) in first_chunk: dense_art[doc].append(first_chunk[(doc, m.group(1).lower())][0])
        docs = []
        for doc in fused:
            chunks = list(dict.fromkeys(dense_art.get(doc, []) + (bm_chunks[doc]['chunks'] if doc in bm_chunks else [])))[:3]
            if chunks: docs.append({'doc_id': doc, 'score': round(sc[doc], 6), 'chunks': chunks, 'chunk_scores': [sc[doc]] * len(chunks)})
        out.write(json.dumps({'qid': qid, 'seg': row.get('seg', ''), 'docs': docs}, ensure_ascii=False) + '\n')
print('ok')
