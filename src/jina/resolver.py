"""Xác định văn bản corpus mà đáp án nền đang trích (từ số hiệu + tiêu đề trong dòng "Căn cứ"; mơ hồ thì dò 3 đoạn thân trong passage)."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fusion'))
import fix_header as fh
_passages={}
def passage(d):
    if d not in _passages:
        _passages[d] = ' '.join(json.load(open(fh.CORP[d]['path']))['passage'].split())
    return _passages[d]

def resolve(a):
    h = next((l for l in a.splitlines()[:2] if l.startswith('Căn cứ')), '')
    dns = fh.DOCNUM.findall(h)
    dn = dns[0] if dns else None
    hn = h.replace(dn + ' ', '', 1) if dn else h
    mt = fh._TITLE_RE.search(hn)
    title = mt.group() if mt else None
    candidates = fh.BY_DN.get(dn, []) if dn else []
    candidates = [d for d in candidates if fh.CORP[d]['title'] == title] or (fh.BY_TITLE.get(title, []) if title else []) or candidates
    if len(candidates) == 1:
        return candidates[0]
    # Resolve ambiguous metadata using only the already emitted body.
    lines = [l.removeprefix('- ') if hasattr(l, 'removeprefix') else (l[2:] if l.startswith('- ') else l)
             for l in a.splitlines()[3:] if len(l.split()) >= 20]
    probes = [' '.join(l.split()[3:23]) for l in lines[:3]]
    scores = {d: sum(p in passage(d) for p in probes) for d in candidates}
    if scores:
        best = max(scores.values())
        winners = [d for d, s in scores.items() if s == best]
        if best >= 2 and len(winners) == 1:
            return winners[0]
    return None
