"""Vector câu hỏi TRAIN (Vietnamese_Embedding, fp16, chuẩn hoá L2) theo thứ tự sorted(qid)
-> WORK/fusion/build/trainq_emb.npy + trainq_ids.json. Dùng cho memory replay và túi từ kNN trong tail.py.

    python3 build_trainq_emb.py [--out WORK/fusion/build]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

from config import BUILD, DENSE_MODEL, TRAIN_JSON   # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(BUILD))
args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

tr = json.loads(Path(TRAIN_JSON).read_text(encoding='utf-8'))
ids = sorted(tr)
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer(DENSE_MODEL, device='cuda'); enc.half()
V = enc.encode([str(tr[q]['question']) for q in ids], batch_size=64,
               normalize_embeddings=True, show_progress_bar=False)
np.save(out / 'trainq_emb.npy', V.astype(np.float16))
(out / 'trainq_ids.json').write_text(json.dumps(ids))
print('trainq_emb', V.shape, '->', out)
