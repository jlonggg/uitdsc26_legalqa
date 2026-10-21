#!/usr/bin/env python3
"""Kiểm tuân thủ thể lệ BTC cho hệ thống nộp bài (CPU, vài giây):
  C  tổng tham số mọi model có trọng số trong models/ (đếm từ header .safetensors, mỗi model 1 lần) < 4 tỷ
  E  mọi model thuộc danh mục BTC phê duyệt; code chỉ nạp đúng 4 model này
  D  không gọi API/HTTP ra ngoài (chỉ Elasticsearch cục bộ)
usage: python3 check_compliance.py"""
import json, re, struct, sys
from math import prod
from pathlib import Path

REPO = Path(__file__).resolve().parent
APPROVED = {   # thư mục models/ -> dòng trong danh mục BTC
    'Vietnamese_Embedding': 'https://huggingface.co/AITeamVN/Vietnamese_Embedding',
    'Vietnamese_Reranker': 'https://huggingface.co/AITeamVN/Vietnamese_Reranker',
    'Qwen3-Embedding-0.6B': 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B',
    'jina-reranker-v3.5': 'https://huggingface.co/jinaai/jina-reranker-v3.5',
}
EXPECTED = {'Vietnamese_Embedding': 567_754_752, 'Vietnamese_Reranker': 567_755_777,
            'Qwen3-Embedding-0.6B': 595_776_512, 'jina-reranker-v3.5': 596_836_352}


def params(model_dir):
    n = 0
    for f in sorted(model_dir.glob('*.safetensors')):
        with open(f, 'rb') as fh:
            head = json.loads(fh.read(struct.unpack('<Q', fh.read(8))[0]))
        n += sum(prod(v['shape']) for k, v in head.items() if k != '__metadata__')
    return n


dirs = sorted(d.name for d in (REPO / 'models').iterdir() if d.is_dir())
assert set(dirs) == set(APPROVED), f'models/ phải chứa đúng 4 model v20, có {dirs}'
total = 0
for d in dirs:
    n = params(REPO / 'models' / d); total += n
    print(f'{d:24s} {n:>15,d}  {APPROVED[d]}')
    assert n == EXPECTED[d], f'{d}: {n} tham số, kỳ vọng {EXPECTED[d]}'
print(f'{"TỔNG":24s} {total:>15,d}  = {total / 4e9:.1%} trần 4.000.000.000')
assert total < 4_000_000_000

src = {p: p.read_text(encoding='utf-8') for p in [*(REPO / 'src').rglob('*.py'), REPO / 'env/lib.sh', REPO / 'run_from_scratch.sh']}
loads = sorted({m for s in src.values() for m in re.findall(r"models/([\w.\-]+)|MODELS\s*/\s*\"([\w.\-]+)\"", s) for m in m if m}
               | {m for s in src.values() for m in re.findall(r'"(Vietnamese_\w+|Qwen3-Embedding-0\.6B|jina-reranker-v3\.5)"', s)})
print('model được code tham chiếu:', loads)
assert set(loads) <= set(APPROVED), 'code tham chiếu model ngoài 4 model v20'
bad = [(str(p.relative_to(REPO)), w) for p, s in src.items()
       for w in ('requests', 'urllib', 'openai', 'anthropic', 'genai', 'api_key', 'Bearer', 'https://') if w in s]
print('từ khoá API/HTTP ngoài trong src/ và script chạy:', bad or 'không có')
assert not bad
print('ĐẠT: 4 model trong danh mục BTC, tổng tham số < 4 tỷ, không API ngoài (ES chạy cục bộ 127.0.0.1)')
