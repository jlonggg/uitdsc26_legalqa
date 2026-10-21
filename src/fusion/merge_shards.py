"""Gộp pred/margins từ các shard của predict.py --shard i/n về đúng thứ tự qid của TEST_JSON.

usage: python3 merge_shards.py <out_dir> <n_shards>
"""
import json, sys
from pathlib import Path

from config import TEST_JSON   # noqa: E402

O = Path(sys.argv[1]); n = int(sys.argv[2])
qids = sorted(json.loads(TEST_JSON.read_text(encoding='utf-8')))

for name, dst in (('pred', 'private_pred.json'), ('margins', 'private_margins.json')):
    merged = {}
    for i in range(n):
        merged.update(json.loads((O / f'{name}_shard{i}.json').read_text(encoding='utf-8')))
    if name == 'pred':
        assert set(merged) == set(qids), f'pred: thiếu {len(set(qids) - set(merged))} qid'
    else:   # câu memory-replay không qua retrieval nên không có margin
        assert set(merged) <= set(qids), 'margins: có qid lạ'
    (O / dst).write_text(json.dumps({q: merged[q] for q in qids if q in merged}, ensure_ascii=False), encoding='utf-8')
    print(name, '->', dst, len(merged))
