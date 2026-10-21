#!/usr/bin/env python3
"""So hai bản nộp (vd hai lần chạy run_from_scratch.sh trên máy khác nhau):
  (1) chênh lệch đáp án: số câu khác, METEOR/ROUGE-L của bản mới khi coi bản tham chiếu là "đáp án" (1.0 = y hệt);
  (2) điểm trên các câu private CÓ gold công khai: 73 câu trùng qid/câu hỏi warmup + câu trùng nguyên văn câu hỏi train
      (BTC không chấm các câu này ở private, nhưng là mẫu duy nhất có nhãn) — chấm cả hai bản, cùng scorer.
Scorer: METEOR nltk (meteor_score, token = str.split()), ROUGE-L F (rouge_score) — theo mô tả score_QA của BTC.
Không nằm trong run_from_scratch.sh; chạy trên host, cần cài nltk + rouge_score (không có trong 2 image).
usage: compare.py <new.zip> <reference.zip>"""
import json, sys, zipfile
from pathlib import Path
from multiprocessing import Pool
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

REPO = Path(__file__).resolve().parent
RS = rouge_scorer.RougeScorer(['rougeL'])
load = lambda p: json.loads(zipfile.ZipFile(p).read('submission.json'))
n = lambda s: ' '.join(str(s).split())


def parts(job):
    hyp, ref = job
    return meteor_score([ref.split()], hyp.split()), RS.score(ref, hyp)['rougeL'].fmeasure


def score(pairs):
    with Pool(16) as p: r = p.map(parts, pairs, chunksize=8)
    return sum(m for m, _ in r) / len(r), sum(x for _, x in r) / len(r), r


new, ref = load(sys.argv[1]), load(sys.argv[2])
P = json.loads((REPO / 'data/private-official_QA.json').read_text(encoding='utf-8'))
W = json.loads((REPO / 'data/LegalQA_warmup.json').read_text(encoding='utf-8'))
T = json.loads((REPO / 'data/train_QA.json').read_text(encoding='utf-8'))
qs = list(P)
diff = [q for q in qs if new[q]['answer'] != ref[q]['answer']]
m, r, _ = score([(new[q]['answer'], ref[q]['answer']) for q in diff]) if diff else (1.0, 1.0, [])
print(f'(1) câu khác bản đã nộp: {len(diff)}/{len(qs)} ({len(diff) / len(qs):.1%})'
      + (f' | trên các câu khác: METEOR(mới, đã nộp) {m:.4f}, ROUGE-L {r:.4f}' if diff else ' — TRÙNG TOÀN BỘ'))
if diff:
    words = lambda d, qq: sum(len(d[q]['answer'].split()) for q in qq) / len(qq)
    print(f'    độ dài TB trên câu khác: mới {words(new, diff):.0f} từ, đã nộp {words(ref, diff):.0f} từ')

tq = {n(v['question']): v['answer'] for v in T.values()}
gold = {}
for q in qs:
    if q in W and n(W[q]['question']) == n(P[q]['question']) and W[q].get('answer'): gold[q] = W[q]['answer']
    elif n(P[q]['question']) in tq: gold[q] = tq[n(P[q]['question'])]
g = sorted(gold)
mn, rn, rows_n = score([(new[q]['answer'], gold[q]) for q in g])
mr, rr, rows_r = score([(ref[q]['answer'], gold[q]) for q in g])
changed = [i for i, q in enumerate(g) if new[q]['answer'] != ref[q]['answer']]
print(f'(2) {len(g)} câu private có gold công khai (warmup/train): METEOR mới {mn:.5f} vs đã nộp {mr:.5f} (Δ {mn - mr:+.5f}) | '
      f'ROUGE-L {rn:.5f} vs {rr:.5f} | {len(changed)} câu trong nhóm này khác nhau')
