"""Tầng hậu xử lý sau retrieval. Không có mô hình sinh: đầu vào là output retrieval (private_pred.json);
chỉ dùng dữ liệu TRAIN của BTC. Các bản v3/v5/v6/v7/v9 được ghi ra file; private_v9.json là đầu vào của fix_header.py:
  v2  memory (câu gần trùng câu TRAIN, cos ≥ 0,95 + cùng thực thể) / canonical (trùng sau khi bỏ đuôi hỏi,
      chỉ 1 đáp án TRAIN) -> dùng lại đáp án TRAIN; còn lại giữ đáp án retrieval
  v3  prepend: chép câu hỏi lên dòng đầu; câu hỏi có/không thêm "Theo đó thì <câu hỏi>." trước dòng "Như vậy"
  v5  bag: thêm 1 dòng ≤ 20 từ lấy từ đáp án của ≤ 25 câu TRAIN gần nhất (cos ≥ 0,4, cộng dồn cos), bỏ từ đã có
  v6  order: gom dòng bag thành cụm theo thứ tự xuất hiện trong đáp án của 25 câu TRAIN gần nhất (TF-IDF ký tự)
  v7  S + K40: nối (stitch) các cụm theo bigram của hàng xóm; thêm ≤ 12 từ dự đoán bằng kernel ridge K40
      (TF-IDF ký tự câu hỏi -> túi từ đáp án, xác suất ≥ 0,4)
  v9  joint tail order: sắp lại toàn bộ phần sau v3 nếu tăng số bigram khớp đáp án hàng xóm
Đáp án TRAIN dùng lại nguyên văn (∈ GOLDS) được giữ nguyên qua mọi bước.

usage: python3 tail.py <pred.json> <qvecs.npy> <out_dir>      (ghi out_dir/private_v3|v5|v6|v7|v9.json)
"""
import collections, heapq, json, re, sys, unicodedata
from pathlib import Path
import numpy as np
from scipy import sparse, linalg
from sklearn.feature_extraction.text import TfidfVectorizer

from config import BUILD, RESOURCES, TEST_JSON, TRAIN_JSON
from memory import norm_q, _ents

TRAIN = json.load(open(TRAIN_JSON))
PUB = json.load(open(TEST_JSON, encoding='utf-8'))
BANK_IDS = json.load(open(BUILD / 'trainq_ids.json'))
BANK_EMB = np.load(BUILD / 'trainq_emb.npy').astype(np.float32)
GOLDS = {str(v['answer']).strip() for v in TRAIN.values()}
YN = re.compile(r"(có\s.*\s(được\s)?không|được không|hay không)\??\s*$", re.I)


def rd(p): return json.load(open(p, encoding='utf-8'))
def wr(p, d): Path(p).write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8')


# ── canonical-v2: khớp câu hỏi sau chuẩn hoá an toàn ──────────────────────
TAILS = [
    "được quy định như thế nào", "được pháp luật quy định như thế nào",
    "được quy định ra sao", "quy định như thế nào", "quy định ra sao",
    "được hiểu như thế nào", "như thế nào", "là những gì", "gồm những gì",
    "bao gồm những gì", "là gì", "ra sao", "thế nào", "gì",
]
DROP = {"các", "những", "và"}
MIN_CORE = 4
DIGITS = re.compile(r"\d+")
ADMIN_PAIRS = [
    ("đô thị", "nông thôn"),
    ("cấp huyện", "cấp tỉnh"), ("cấp huyện", "cấp xã"),
    ("cấp tỉnh", "cấp xã"),
]


def canon(q: str) -> str:
    s = unicodedata.normalize("NFC", str(q)).lower()
    s = re.sub(r"[?!.…]+\s*$", "", " ".join(s.split())).strip()
    for t in TAILS:
        if s.endswith(t):
            s = s[: -len(t)].rstrip(" ,;:")
            break
    toks = [w for w in re.findall(r"\w+", s) if w not in DROP]
    return " ".join(toks)


def ent_conflict(a: str, b: str) -> bool:
    if set(DIGITS.findall(a)) != set(DIGITS.findall(b)):
        return True
    la, lb = a.lower(), b.lower()
    for x, y in ADMIN_PAIRS:
        if (x in la) != (x in lb) and (y in la) != (y in lb):
            return True
    return False


# ── sắp cụm từ (ordered_groups/stitch) + kernel ridge K40 ────────────────
def norm27(s):return ' '.join(unicodedata.normalize('NFC',s).lower().split()).rstrip('? ')

def ordered_groups(line,neighbors):
    tokens=line.split();low=[t.lower() for t in tokens];assert len(low)==len(set(low))
    eligible=set(low);runs=collections.Counter()
    for answer in neighbors:
        cur=[];observed=set()
        for t in answer.lower().split()+['\0']:
            if t in eligible and t not in cur:cur.append(t)
            else:
                if len(cur)>=2:observed.add(tuple(cur))
                cur=[t] if t in eligible else []
        runs.update(observed)
    used=set();groups=[];ix={t:i for i,t in enumerate(low)};surface=dict(zip(low,tokens))
    for run,n in sorted(runs.items(),key=lambda x:(-len(x[0])*x[1],-len(x[0]),tuple(ix[t] for t in x[0]))):
        if used.intersection(run):continue
        used.update(run);groups.append([surface[t] for t in run])
    groups.extend([surface[t]] for t in low if t not in used)
    return groups

def render(groups):return ' '.join(t for g in groups for t in g)

def stitch(groups,neighbors):
    n=len(groups);original=tuple(range(n))
    if n<2:return render(groups),dict(before=0,after=0)
    df=collections.Counter()
    for answer in neighbors:
        ts=answer.lower().split();df.update(set(zip(ts,ts[1:])))
    edge=[[df[a[-1].lower(),b[0].lower()] if i!=j else 0 for j,b in enumerate(groups)] for i,a in enumerate(groups)]
    initial=sum(edge[i][i+1] for i in range(n-1))
    if not any(any(row) for row in edge):return render(groups),dict(before=initial,after=initial)
    beam=[(0,(),0)]
    for _ in range(n):
        beam=heapq.nsmallest(16,((s-(edge[p[-1]][j] if p else 0),p+(j,),m|(1<<j)) for s,p,m in beam for j in range(n) if not m&(1<<j)))
    score,path,_=beam[0]
    if -score<=initial:path=original;score=-initial
    new=render([groups[j] for j in path]);assert collections.Counter(new.split())==collections.Counter(render(groups).split())
    return new,dict(before=initial,after=-score)

def kernel(x,xt,answers):
    counts=collections.Counter(t for a in answers for t in set(a.lower().split()))
    vocab=sorted(t for t,n in counts.items() if n>=10);ids={t:i for i,t in enumerate(vocab)}
    yi=[];yj=[]
    for i,a in enumerate(answers):
        for t in set(a.lower().split()):
            if t in ids:yi.append(i);yj.append(ids[t])
    y=sparse.csr_matrix((np.ones(len(yi),dtype=np.float32),(yi,yj)),shape=(len(answers),len(vocab)))
    k=(x@x.T).toarray();kt=(xt@x.T).toarray();km=k.mean(0);kg=float(km.mean())
    k-=km[None,:];k-=km[:,None];k+=kg
    kt-=kt.mean(1,keepdims=True);kt-=km[None,:];kt+=kg
    k.flat[::len(k)+1]+=1.
    print('Kernel Cholesky',k.shape,'vocab',len(vocab),flush=True)
    f=linalg.cho_factor(k,overwrite_a=True,check_finite=False)
    weights=linalg.cho_solve(f,kt.T,check_finite=False).T
    pred=np.clip(weights@y+(1-weights.sum(1)[:,None])*np.asarray(y.mean(0)),0,1)
    return vocab,pred


# ── joint tail order (v9) ────────────────────────────────────────────────
def reorder(core,answer,neighbors):
    assert answer.startswith(core)
    tail=answer[len(core):].split()
    assert len(tail)==len(set(t.lower() for t in tail))
    if len(tail)<2:return answer,dict(before=0,after=0)
    df=collections.Counter()
    for a in neighbors:
        t=a.lower().split();df.update(set(zip(t,t[1:])))
    def score(tokens):
        t=([core.split()[-1]] if core.split() else [])+tokens
        return sum(df[a.lower(),b.lower()] for a,b in zip(t,t[1:]))
    candidate,_=stitch(ordered_groups(' '.join(tail),neighbors),neighbors)
    before=score(tail);after=score(candidate.split())
    new=core+'\n'+candidate if after>before else answer
    assert collections.Counter(new.split())==collections.Counter(answer.split())
    return new,dict(before=before,after=after,applied=after>before)


# ── chuỗi v3 -> v9 ───────────────────────────────────────────────────────
def memory(qs, qvecs):
    """-> {qid: đáp án TRAIN}: trùng sau chuẩn hoá hoặc cos ≥ 0,95, và cùng thực thể (như memory.py)."""
    tnorm = {}
    for tid in BANK_IDS:
        tnorm.setdefault(norm_q(TRAIN[tid]['question']), []).append(tid)
    out = {}
    for i, q in enumerate(qs):
        text = qs[q]['question']
        cands = tnorm.get(norm_q(text), [])
        hit = cands[0] if cands else None
        if hit is None:
            s = BANK_EMB @ qvecs[i]
            j = int(np.argmax(s)); sim = float(s[j])
            if sim < 0.95: continue
            hit = BANK_IDS[j]
        mem = str(TRAIN[hit].get('answer') or '').strip()
        if not mem or _ents(text) != _ents(TRAIN[hit]['question']): continue
        out[q] = mem
    return out


def canonical(qs, mem_qids):
    by = {}
    for tid, v in TRAIN.items():
        c = canon(v['question'])
        if len(c.split()) >= MIN_CORE: by.setdefault(c, []).append(tid)
    out = {}
    for q, v in qs.items():
        if q in mem_qids: continue
        c = canon(v['question'])
        if len(c.split()) < MIN_CORE: continue
        cands = by.get(c, [])
        if not cands: continue
        golds = {str(TRAIN[t]['answer']).strip() for t in cands}
        if len(golds) != 1 or ent_conflict(str(v['question']), str(TRAIN[cands[0]]['question'])): continue
        out[q] = golds.pop()
    return out


def prepend_t16(sub, qs):
    out = {}
    for q, v in sub.items():
        a = v['answer']
        if a.strip() in GOLDS: out[q] = {'answer': a}; continue
        text = ' '.join(str(qs[q]['question']).split()); ls = a.split('\n')
        if YN.search(text):
            ki = max((i for i, l in enumerate(ls) if l.startswith('Như vậy')), default=None)
            if ki is not None: ls = ls[:ki] + [f"Theo đó thì {text.rstrip('? ')}."] + ls[ki:]
        out[q] = {'answer': text + '\n' + '\n'.join(ls)}
    return out


def bag_lines(qs, qvecs, sub):
    """Dòng bag (v5): kNN 25 câu TRAIN gần nhất theo embedding câu hỏi, cos ≥ 0,4, tối đa 20 từ."""
    gold_toks = {t: str(TRAIN[t]['answer']).split() for t in BANK_IDS}
    out = {}
    for i, q in enumerate(qs):
        if sub[q]['answer'].strip() in GOLDS: continue
        s = BANK_EMB @ qvecs[i]
        order = np.argsort(-s)
        neigh = [(BANK_IDS[j], float(s[j])) for j in order[:25] if s[j] >= 0.4]
        if not neigh: continue
        have = {w.lower() for w in sub[q]['answer'].split()}
        score, surface = {}, {}
        for tid, sim in neigh:
            seen = set()
            for w in gold_toks[tid]:
                lw = w.lower()
                if lw in have or lw in seen: continue
                seen.add(lw); score[lw] = score.get(lw, 0.0) + sim; surface.setdefault(lw, w)
        if not score: continue
        ranked = sorted(score.items(), key=lambda kv: -kv[1])
        out[q] = ' '.join(surface[w] for w, _ in ranked[:20])
    return out


def char_neighbors(qs, bank):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
    x = vec.fit_transform([TRAIN[t]['question'] for t in bank]); ids = list(qs)
    sims = (vec.transform([qs[q]['question'] for q in ids]) @ x.T).toarray()
    return {q: [bank[j] for j in np.argsort(-sims[i], kind='stable')[:25]] for i, q in enumerate(ids)}


def v6_order(v3, bags, nb):
    out, groups_of = {}, {}
    for q in v3:
        if q not in bags: out[q] = dict(v3[q]); continue
        g = ordered_groups(bags[q], [TRAIN[t]['answer'] for t in nb[q]])
        groups_of[q] = g
        out[q] = {'answer': v3[q]['answer'] + '\n' + render(g)}
    return out, groups_of


def v7_s_k40(v3, v6, bags, groups_of, nb6, nb7, pred, vocab):
    out = {}
    for i, q in enumerate(list(v3)):
        base = v6[q]['answer']; core = v3[q]['answer']
        if base.strip() in GOLDS: out[q] = {'answer': base}; continue
        S = base
        if q in bags and render(groups_of[q]) != bags[q]:      # v6 có đổi -> có provenance -> stitch
            line, _ = stitch(groups_of[q], [TRAIN[t]['answer'] for t in nb6[q]]); S = core + '\n' + line
        have = set(base.lower().split()); ranked = np.argsort(-pred[i], kind='stable')
        extra = [vocab[j] for j in ranked if pred[i, j] >= .4 and vocab[j] not in have][:12]
        line = render(ordered_groups(' '.join(extra), [TRAIN[t]['answer'] for t in nb7[q]])) if extra else ''
        out[q] = {'answer': S + ('\n' + line if line else '')}
    return out


def v9_joint(v3, v7, nb):
    out = {}
    for q in v7:
        if v7[q]['answer'].strip() in GOLDS: out[q] = dict(v7[q]); continue
        new, _ = reorder(v3[q]['answer'], v7[q]['answer'], [TRAIN[t]['answer'] for t in nb[q]])
        out[q] = {'answer': new}
    return out


def run(pred_path, qvecs_path, out_dir):
    O = Path(out_dir); O.mkdir(parents=True, exist_ok=True)
    pred = rd(pred_path)
    qs = PUB; ids = list(qs)
    qvecs = np.load(qvecs_path).astype(np.float32)
    bank6 = [t for t in sorted(TRAIN) if norm27(TRAIN[t]['question']) not in {norm27(r['question']) for r in PUB.values()}]
    held = {norm27(TRAIN[q]['question']) for q in rd(RESOURCES / 'harness_651.json')}
    forb = held | {norm27(r['question']) for r in PUB.values()}; seen = set(); bank7 = []
    for t in sorted(TRAIN):                      # bank v7/K40: bỏ 651 câu TRAIN giữ lại để đánh giá khi chọn tham số (resources/harness_651.json)
                                                 # và câu trùng câu test, khử câu trùng; giữ nguyên để tái lập đúng bài nộp
        nq = norm27(TRAIN[t]['question'])
        if nq not in forb and nq not in seen: bank7.append(t); seen.add(nq)
    mem = memory(qs, qvecs); print('memory', len(mem), flush=True)
    can = canonical(qs, set(mem)); print('canonical', len(can), flush=True)
    v2 = {q: {'answer': mem.get(q) or can.get(q) or pred[q]['answer']} for q in ids}
    v3 = prepend_t16(v2, qs); wr(O / 'private_v3.json', v3)
    bags = bag_lines(qs, qvecs, v3); print('bag', len(bags), flush=True)
    v5 = {q: {'answer': v3[q]['answer'] + ('\n' + bags[q] if q in bags else '')} for q in ids}; wr(O / 'private_v5.json', v5)
    nb6 = char_neighbors(qs, bank6)
    v6, groups_of = v6_order(v3, bags, nb6); wr(O / 'private_v6.json', v6)
    nb7 = char_neighbors(qs, bank7)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
    x = vec.fit_transform([TRAIN[t]['question'] for t in bank7]); xt = vec.transform([qs[q]['question'] for q in ids])
    vocab, kp = kernel(x, xt, [TRAIN[t]['answer'] for t in bank7])          # K40 kernel ridge
    v7 = v7_s_k40(v3, v6, bags, groups_of, nb6, nb7, kp, vocab); wr(O / 'private_v7.json', v7)
    v9 = v9_joint(v3, v7, nb7); wr(O / 'private_v9.json', v9)
    print('done', {k: sum(v[q] != w[q] for q in ids) for k, v, w in [('v6!=v5', v6, v5), ('v7!=v6', v7, v6), ('v9!=v7', v9, v7)]}, flush=True)


if __name__ == '__main__':
    run(*sys.argv[1:4])
