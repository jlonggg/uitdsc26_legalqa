#!/usr/bin/env python3
"""Reorder a pool by cross-encoder logits: docs by max chunk logit, chunks by logit; keep logits as chunk_scores.
usage: ce_pool.py <ce.jsonl> <pool_in.jsonl> <pool_out.jsonl>"""
import json, sys
ce = {r['qid']: r for r in map(json.loads, open(sys.argv[1], encoding='utf-8'))}
with open(sys.argv[3], 'w', encoding='utf-8') as out:
    for l in open(sys.argv[2], encoding='utf-8'):
        row = json.loads(l); c = ce.get(row['qid'])
        if not c: continue
        docs = []
        for d in c['docs']:
            pairs = sorted(zip(d['logits'], d['chunks']), reverse=True)
            docs.append({'doc_id': d['doc_id'], 'score': pairs[0][0], 'chunks': [p[1] for p in pairs], 'chunk_scores': [p[0] for p in pairs]})
        docs.sort(key=lambda d: -d['score'])
        out.write(json.dumps({'qid': row['qid'], 'seg': row.get('seg', ''), 'docs': docs}, ensure_ascii=False) + '\n')
print('ok')
