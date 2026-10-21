#!/usr/bin/env python3
"""Hậu xử lý dòng "Căn cứ" của đáp án (không dùng nhãn):
  (a) law: header mang quốc hiệu thay tiêu đề (văn bản Quốc hội, số hiệu đuôi QH) -> tên luật + năm;
  (b) own: số hiệu trong header là số hiệu tham chiếu -> số hiệu tự thân (dòng "Số:" của văn bản) + 'năm YYYY';
  (c) own_nodn: header không có số hiệu -> chèn số hiệu tự thân + năm sau loại văn bản trong tiêu đề.
Không chạy lại retrieval: văn bản được suy từ (số hiệu, tiêu đề) có trong dòng "Căn cứ"; chỉ đổi khi ánh xạ về corpus là duy nhất.
usage: fix_header.py <in.json> <out.json>      (pipeline: private_v9.json -> private_v11.json)"""
import json, re, sys
from pathlib import Path
from corpus import load_corpus, load_passage   # noqa: E402
CORP = {d: dict(v, path=str(v['path'])) for d, v in load_corpus().items()}
_OWN_RE = re.compile(r'[Ss]ố:\s*(\d+/[\w\-/]+)')   # số hiệu "Số:" tự thân trong 1.500 ký tự đầu
OWN = {d: (lambda m: m.group(1) if m else '')(_OWN_RE.search(load_passage(v['path'])[:1500])) for d, v in CORP.items()}
MOTTO = 'CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM'
DOCNUM = re.compile(r"\b(\d{1,4}/\d{4}/[A-ZĐ][\w\-ĐđĂăÂâÊêÔôƠơƯư]*(?:-[A-ZĐ][\w]*)*)\b")
BY_DN = {}
for d, v in CORP.items():
    if v['docnums']: BY_DN.setdefault(v['docnums'][0], []).append(d)
_head = {}
def head_of(doc):
    if doc not in _head:
        _head[doc] = ' '.join(json.load(open(CORP[doc]['path']))['passage'][:2500].split())
    return _head[doc]
def year_of(doc):
    m = re.search(r'ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+(\d{4})', head_of(doc)); return m.group(1) if m else None
def law_name(doc):
    p = head_of(doc); m = re.search(r'năm\s+(\d{4})\s+', p)
    if not m: return None
    toks = p[m.end():].split()
    if not toks or toks[0] not in ('BỘ', 'LUẬT', 'PHÁP', 'NGHỊ'): return None
    run = []
    for t in toks[:14]:
        core = re.sub(r'[^\wĐđ]', '', t)
        if core and core.isupper(): run.append(t)
        else: break
    if len(run) < 2: return None
    k = 2 if run[0] in ('BỘ', 'PHÁP', 'NGHỊ') else 1
    rest = ' '.join(run[k:]).lower(); rest = rest[:1].upper() + rest[1:]
    return ' '.join(t.capitalize() for t in run[:k]) + ' ' + rest + ' ' + m.group(1)
BY_TITLE = {}
for d, v in CORP.items():
    t = v['title']
    if v['title_src'] == 'passage' and MOTTO not in t and len(t.split()) >= 3: BY_TITLE.setdefault(t, []).append(d)
_TITLE_RE = re.compile('|'.join(re.escape(t) for t in sorted(BY_TITLE, key=len, reverse=True)))
DOCTYPES = ["thông tư liên tịch", "bộ luật", "thông tư", "nghị định", "nghị quyết", "quyết định", "chỉ thị",
            "pháp lệnh", "hiến pháp", "công văn", "luật"]
def fix_line(h):
    dn = DOCNUM.findall(h); dn = dn[0] if dn else None
    h_nodn = h.replace(dn + ' ', '', 1) if dn else h
    m = _TITLE_RE.search(h_nodn); title = m.group(0) if m else None
    if dn:
        cands = BY_DN.get(dn, [])
        c2 = [d for d in cands if CORP[d]['title'] == title] or (BY_TITLE.get(title, []) if title else []) or cands
    else:
        c2 = BY_TITLE.get(title, []) if title else []
        keys = {(OWN.get(d, ''), year_of(d)) for d in c2}
        if len(keys) != 1: return h, None
        own, y = keys.pop()
        if not own or not y or DOCNUM.fullmatch(own): return h, None
        low = title.lower(); k = next((len(t.split()) for t in DOCTYPES if low.startswith(t)), 0)
        if not k: return h, None
        tt = title.split(); new = ' '.join(tt[:k] + [f'{own} năm {y}'] + tt[k:])
        return h.replace(title, new, 1), 'own_nodn'
    if not c2: return h, None
    if MOTTO in h:
        if not re.search(r'QH\d+$', dn): return h, None
        names = {law_name(d) for d in c2} - {None}
        if len(names) != 1: return h, None
        return h.replace(dn + ' ' + MOTTO, names.pop()), 'law'
    keys = {(OWN.get(d, ''), year_of(d)) for d in c2}
    if len(keys) != 1: return h, None
    own, y = keys.pop()
    if not own or own == dn or not y or DOCNUM.fullmatch(own): return h, None
    return h.replace(dn, f'{own} năm {y}', 1), 'own'
def fix_answer(a):
    ls = a.split('\n'); i = next((i for i, l in enumerate(ls[:2]) if l.startswith('Căn cứ')), None)
    if i is None: return a, None
    ls[i], kind = fix_line(ls[i]); return '\n'.join(ls), kind
if __name__ == '__main__':
    src = json.load(open(sys.argv[1], encoding='utf-8')); out = {}; kinds = {}
    for q, v in src.items():
        a, k = fix_answer(v['answer']); out[q] = {'answer': a}
        if k: kinds[k] = kinds.get(k, 0) + 1
    Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print('đổi', kinds, 'trên', len(src))
