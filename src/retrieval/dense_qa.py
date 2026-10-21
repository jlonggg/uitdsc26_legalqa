#!/usr/bin/env python3
"""Zero-shot Qwen3-Embedding-0.6B retrieval for QA questions over the article vectors of dense_qwen3.py (qwen3_zero.npy); no labels used.
Chạy trong image GPU (env/Dockerfile.gpu): python3 dense_qa.py <questions.json> <out.jsonl>
Output per qid: top-100 docs by MaxP and top-60 article units (doc_id, article, unit_id, score)."""
import json, sys, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import dense_qwen3 as D
qpath, opath = sys.argv[1], sys.argv[2]
rows = D.meta(); docs = [r['doc_id'] for r in rows]
order = list(dict.fromkeys(docs)); dnum = {d: i for i, d in enumerate(order)}
pdoc = torch.tensor([dnum[d] for d in docs]).cuda()
vec = torch.from_numpy(np.load(D.VEC, mmap_mode='r')[:]).cuda()          # fp16, 192670 x 1024 (~400MB)
tok, model = D.load()
qs = json.load(open(qpath, encoding='utf-8')); qids = list(map(str, qs))
with open(opath, 'w', encoding='utf-8') as out:
    for s in range(0, len(qids), 16):
        block = qids[s:s+16]
        q = D.encode(tok, model, [str(qs[k]['question']) for k in block], True, D.MAX_Q).half()
        ps = q @ vec.T                                                     # 16 x 192670
        aval, aind = torch.topk(ps, 60, dim=1)
        dscores = torch.full((len(block), len(order)), -1e4, device='cuda', dtype=ps.dtype)
        dscores.scatter_reduce_(1, pdoc[None].expand(len(block), -1), ps, reduce='amax', include_self=True)
        dval, dind = torch.topk(dscores, 100, dim=1)
        for i, k in enumerate(block):
            out.write(json.dumps({'qid': k,
                'docs': [[order[int(j)], round(float(v), 4)] for j, v in zip(dind[i], dval[i])],
                'arts': [[rows[int(j)]['doc_id'], rows[int(j)]['article'][:12], rows[int(j)]['unit_id'], round(float(v), 4)] for j, v in zip(aind[i], aval[i])]}, ensure_ascii=False) + '\n')
        if (s // 16) % 50 == 0: print(f'{s+len(block)}/{len(qids)}', flush=True)
print('done')
