"""Dựng bản nộp cuối: nền `private_v11.json` + tối đa 2 dòng evidence từ các ứng viên cấp Điều Jina xếp cao nhất
(điểm = trung bình lượt xuôi/ngược), chèn ngay TRƯỚC dòng trích (quote). Mỗi dòng evidence = span 160 từ của ứng viên (bước 40 từ),
bỏ nếu trùng 4-gram ≥ 0,6 với đáp án hiện tại; ứng viên đầu bị bỏ thì không chèn gì.
Span được chọn = cửa sổ có tổng IDF (resources/idf.json) của từ trùng câu hỏi lớn nhất.
usage: compose.py <jina_dir> <out_predictions.json> [span dup topk stride_div]"""
import json, math, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src/fusion'))
from config import RESOURCES, WORK   # noqa: E402


# ── chọn span evidence ──────────────────────────────────────────────────
def tokens(s):return re.findall(r'\w+',s.lower())
def grams(t):return set(tuple(t[i:i+4]) for i in range(len(t)-3))
def bigrams(t):return set(zip(t,t[1:]))
def evidence(question,answer,document,idf,span_n,dup_thr,stride_div=2,mode='idf'):
    words=document['text'].split('\n',1)[1].split();best=None;stride=max(1,span_n//stride_div)
    qt=tokens(question);query=set(qt);qb=bigrams(qt);ref=None
    if mode=='ans':ref=set(tokens(answer))
    for start in range(0,max(1,len(words)-span_n+1),stride):
        span=words[start:start+span_n];st=tokens(' '.join(span));ts=set(st)
        if mode=='idf':k=math.fsum(idf.get(t,1) for t in sorted(query&ts))
        elif mode=='bigram':k=len(qb&bigrams(st))/max(len(qb),1)
        else:k=math.fsum(idf.get(t,1) for t in sorted(ref&ts))
        key=(k,-start)
        if best is None or key>best[0]:best=(key,span,start)
    span=best[1];text=' '.join(span);gg=grams(tokens(text));dup=len(gg&grams(tokens(answer)))/max(len(gg),1)
    if len(span)<16 or dup>=dup_thr:return None
    return document['text'].split('\n',1)[0]+': “'+('… ' if best[2] else '')+text+(' …' if best[2]+len(span)<len(words) else '')+'”'


# ── dựng đáp án ─────────────────────────────────────────────────────────
rd = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
J = Path(sys.argv[1]); out = Path(sys.argv[2])
span, dup, topk, sd = (int(sys.argv[3]) if len(sys.argv) > 3 else 160), (float(sys.argv[4]) if len(sys.argv) > 4 else .6), (int(sys.argv[5]) if len(sys.argv) > 5 else 2), (int(sys.argv[6]) if len(sys.argv) > 6 else 4)
BASE = rd(WORK / 'fusion/private_v11.json'); PRED = rd(WORK / 'fusion/private_pred.json')
IDF = rd(RESOURCES / 'idf.json'); INP = {r['qid']: r for r in rd(J / 'inputs.json')}; R = {}
for p in sorted(J.glob('shard_*/output/scores.json')): R.update(rd(p))
assert set(R) == set(INP) == set(BASE), (len(R), len(INP), len(BASE))
def render(q):   # dòng trích = dòng thứ 4 từ cuối đáp án retrieval (sau nó: câu kết + 2 dòng BOILER_TAIL)
    ans = BASE[q]['answer']; rl = PRED[q]['answer'].split('\n')
    if len(rl) < 4: return ans, 0
    quote = rl[-4]; ls = ans.split('\n'); m = [i for i, l in enumerate(ls) if l == quote]
    if len(m) != 1: return ans, 0
    s = R[q]; order = sorted(s['forward'], key=lambda d: (-(s['forward'][d] + s['reverse'][d]) / 2, d)); ins = []; cur = ans
    for d in order[:topk]:
        doc = next(x for x in INP[q]['docs'] if x['chunk_id'] == d); ln = evidence(INP[q]['question'], cur, doc, IDF, span, dup, sd)
        if ln is None:
            if not ins: break
            continue
        ins.append(ln); cur = cur + '\n' + ln
    return ('\n'.join(ls[:m[0]] + ins + ls[m[0]:]) if ins else ans), len(ins)
pred = {}; n_ins = 0; n_q = 0
for q in BASE:
    a, k = render(q); pred[q] = {'answer': a}; n_ins += k; n_q += k > 0
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(pred, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'params': [span, dup, topk, sd], 'n': len(pred), 'evidence_lines': n_ins, 'q_with_evidence': n_q,
                  'changed_vs_base': sum(pred[q]['answer'] != BASE[q]['answer'] for q in BASE)}, indent=1))
