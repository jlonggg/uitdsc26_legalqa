"""Đóng gói ZIP deterministic (submission.json duy nhất, timestamp 1980) theo thứ tự qid file private gốc. usage: pack.py <predictions.json> <submission.zip>"""
import hashlib, io, json, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fusion'))
from config import TEST_JSON   # noqa: E402
priv = json.loads(TEST_JSON.read_text(encoding='utf-8'))
sub = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert set(sub) == set(priv), 'lệch qid'
out = {q: {'answer': sub[q]['answer']} for q in priv}

raw = json.dumps(out, ensure_ascii=False).encode()
b = io.BytesIO(); z = zipfile.ZipFile(b, 'w')
i = zipfile.ZipInfo('submission.json', (1980, 1, 1, 0, 0, 0))
i.compress_type = zipfile.ZIP_DEFLATED; i.external_attr = 0o600 << 16
z.writestr(i, raw); z.close()
Path(sys.argv[2]).write_bytes(b.getvalue())
print('zip', sys.argv[2], 'sha256', hashlib.sha256(b.getvalue()).hexdigest(), 'n', len(out))
