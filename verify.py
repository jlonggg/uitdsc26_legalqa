#!/usr/bin/env python3
"""Kiểm bản nộp theo định dạng BTC (ZIP chỉ chứa submission.json UTF-8 không BOM, đủ đúng question_id của
private-official_QA.json, answer là chuỗi khác rỗng) và so với bản đã nộp private (official METEOR 0.6433)
qua sha256 từng đáp án trong resources/reference_submission.sha256.json (repo không kèm văn bản bài nộp).
usage: verify.py <submission.zip>"""
import hashlib, json, sys, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()

z = zipfile.ZipFile(sys.argv[1]); names = z.namelist()
assert names == ['submission.json'], f'ZIP phải chứa đúng 1 tệp submission.json, có {names}'
raw = z.read('submission.json'); raw.decode('utf-8')
assert not raw.startswith(b'\xef\xbb\xbf'), 'không được có BOM'
sub = json.loads(raw)
qs = json.loads((REPO / 'data/private-official_QA.json').read_text(encoding='utf-8'))
assert isinstance(sub, dict) and set(sub) == set(qs), 'thiếu/thừa question_id'
assert all(isinstance(v, dict) and isinstance(v.get('answer'), str) and v['answer'].strip() for v in sub.values()), 'answer phải là chuỗi khác rỗng'
print(f'định dạng OK: {len(sub)} câu | zip sha256 {sha(Path(sys.argv[1]).read_bytes())} | submission.json sha256 {sha(raw)}')
ref = json.loads((REPO / 'resources/reference_submission.sha256.json').read_text(encoding='utf-8'))
diff = [q for q in qs if sha(sub[q]['answer'].encode('utf-8')) != ref['answer_sha256'][q]]
print(f'so bản đã nộp (0.6433): submission.json {"TRÙNG BYTE" if sha(raw) == ref["submission_json_sha256"] else "khác"} | '
      f'{len(qs) - len(diff)}/{len(qs)} câu trùng, {len(diff)} câu khác' + (f' (vd {diff[:5]})' if diff else ''))
