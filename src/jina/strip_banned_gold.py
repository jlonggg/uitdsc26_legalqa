"""Gỡ gold khỏi các câu BỊ CẤM (trùng qid hoặc trùng 100% nội dung với warmup/train).

Thay bằng đoạn corpus top-1 của pool CE — cùng dạng "trích nguyên văn đoạn luật" như pipeline.
Các câu chỉ TƯƠNG TỰ (cosine cao nhưng câu hỏi khác chữ) được giữ nguyên.
usage: strip_banned_gold.py <in.json> <out.json>
(warmup CHỈ dùng để nhận diện câu bị loại, không chép đáp án warmup vào bài nộp)
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fusion'))
from config import DATA, TEST_JSON, TRAIN_JSON, WORK   # noqa: E402
rd = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
n = lambda s: ' '.join(str(s).split())

P, W, T = rd(TEST_JSON), rd(DATA / 'LegalQA_warmup.json'), rd(TRAIN_JSON)
sub = rd(sys.argv[1])
wq = {n(r['question']) for r in W.values()}; tq = {n(r['question']) for r in T.values()}
gold = {n(r['answer']) for r in T.values()} | {n(r['answer']) for r in W.values()}
ban = {q for q in P if q in W or q in T or n(P[q]['question']) in wq or n(P[q]['question']) in tq}
bad = [q for q in sub if q in ban and n(sub[q]['answer']) in gold]

pool = {r['qid']: r for r in map(json.loads, open(WORK / 'retrieval/pool_private_ce.jsonl', encoding='utf-8'))}
need = {pool[q]['docs'][0]['chunks'][0] for q in bad}
chunk = {}
for line in open(WORK / 'retrieval/chunks.jsonl', encoding='utf-8'):
    r = json.loads(line)
    if r['id'] in need: chunk[r['id']] = r

out = dict(sub)
for q in bad:
    d = pool[q]['docs'][0]; c = chunk[d['chunks'][0]]
    out[q] = {'answer': f"Căn cứ {c.get('article', '')}:\n{c['text']}".strip()}
assert all(n(out[q]['answer']) not in gold for q in ban), 'còn câu cấm mang gold'
Path(sys.argv[2]).write_text(json.dumps({q: out[q] for q in P}, ensure_ascii=False), encoding='utf-8')
print(f'gỡ gold {len(bad)} câu cấm: {bad}')
