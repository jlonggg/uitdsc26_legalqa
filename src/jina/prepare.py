"""Ứng viên cấp ĐIỀU cho Jina (không dùng nhãn), khoá theo chunk_id, từ pool CE (pool_private_ce.jsonl):
  · K văn bản đầu: M chunk có nhiều bigram trùng câu hỏi nhất ∪ các chunk pool sẵn có (tối đa M+2 chunk);
  · các văn bản còn lại của pool: chunk CE cao nhất;
  · + đoạn trích (quote) của đáp án nền làm ứng viên chunk_id 'incumbent'; văn bản của nó xác định từ
    private_v9.json bằng resolver.py.
Bài nộp dùng K=3, M=8 (trung bình 39 ứng viên/câu, tối đa 43). Kiểm 29/09/2026: inputs.json trùng sha256 bản đã dùng (ad4b2b12…).
usage: prepare.py <out_dir> [K=3] [M=8] [NSHARD=3]  -> out_dir/inputs.json + hash.json, out_dir/shard_i/input/ = rows[i::NSHARD]"""
import hashlib, json, re, sys, collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src/fusion')); sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TEST_JSON, WORK      # noqa: E402
from resolver import resolve, fh        # noqa: E402
out = Path(sys.argv[1]); K, M, NSH = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((2, 3), (3, 8), (4, 3)))
rd = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
questions = rd(TEST_JSON); raw = rd(WORK / 'fusion/private_pred.json'); old = rd(WORK / 'fusion/private_v9.json')
pool_p = WORK / 'retrieval/pool_private_ce.jsonl'
pool = {r['qid']: r for r in map(json.loads, open(pool_p, encoding='utf-8'))}; assert set(questions) == set(pool) == set(raw) == set(old)
topdocs = {d['doc_id'] for r in pool.values() for d in r['docs'][:K]}; fixed = {d['chunks'][0] for r in pool.values() for d in r['docs']} | {c for r in pool.values() for d in r['docs'][:K] for c in d['chunks']}
bydoc = collections.defaultdict(list); chunks = {}
for line in open(WORK / 'retrieval/chunks.jsonl', encoding='utf-8'):
    c = json.loads(line)
    if c['doc_id'] in topdocs: bydoc[c['doc_id']].append(c); chunks[c['id']] = c
    elif c['id'] in fixed: chunks[c['id']] = c
def tokens(s): return re.findall(r'\w+', s.lower())
def bigr(t): return set(zip(t, t[1:]))
def fmt(doc, c, text): return {'doc_id': doc, 'chunk_id': c['id'] if c else 'incumbent', 'text': 'Văn bản ' + fh.OWN.get(doc, '') + (' ' + c.get('article', '') if c and c.get('article') else '') + '\n' + ' '.join(text.split()[:256])}
rows = []; known = 0; sizes = []
for q in sorted(questions):
    qt = tokens(questions[q]['question']); qb = bigr(qt); inc = resolve(old[q]['answer']); known += inc is not None; docs = []; seen = set()
    for i, d in enumerate(pool[q]['docs']):
        doc = d['doc_id']
        if i < K:
            cand = sorted(bydoc[doc], key=lambda c: (-len(qb & bigr(tokens(c['text']))), c.get('order', 0)))[:M]
            ids = list(dict.fromkeys([c['id'] for c in cand] + [x for x in d['chunks'] if x in chunks]))[:M + 2]
        else:
            ids = [d['chunks'][0]] if d['chunks'][0] in chunks else []
        for cid in ids:
            if cid not in seen: seen.add(cid); docs.append(fmt(doc, chunks[cid], chunks[cid]['text']))
    rl = raw[q]['answer'].split('\n')   # rl[-4] = dòng trích của đáp án retrieval
    if inc and len(rl) >= 4: docs.append(fmt(inc, None, rl[-4]))
    sizes.append(len(docs)); rows.append({'qid': q, 'question': questions[q]['question'], 'docs': docs})
p = out / 'inputs.json'
def write(d, sub):
    d.mkdir(parents=True, exist_ok=True); f = d / 'inputs.json'; f.write_text(json.dumps(sub, ensure_ascii=False), encoding='utf-8')
    (d / 'hash.json').write_text(json.dumps({'inputs.json': hashlib.sha256(f.read_bytes()).hexdigest()}))
write(out, rows)
for i in range(NSH): write(out / f'shard_{i}/input', rows[i::NSH]); (out / f'shard_{i}/output').mkdir(exist_ok=True)
import statistics as st
print(json.dumps({'K': K, 'M': M, 'n': len(rows), 'incumbent_resolved': known, 'cands_mean': st.mean(sizes), 'cands_max': max(sizes), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}, indent=1))
