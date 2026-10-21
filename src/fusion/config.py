"""Cấu hình nền v11. Giá trị hằng số giữ NGUYÊN như bản đã nộp (chọn bằng thực nghiệm trên tập train).
Đường dẫn: DATA (dữ liệu BTC), MODELS (4 model), WORK (output) — override được bằng biến môi trường cùng tên.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DATA", str(REPO / "data")))
MODELS = Path(os.environ.get("MODELS", str(REPO / "models")))
WORK = Path(os.environ.get("WORK", str(REPO / "work")))
ART = WORK / "fusion" / "index"                      # chunks.pkl, bm25s_index/, emb_aiteam.npy
BUILD = WORK / "fusion" / "build"                    # trainq_emb.npy, trainq_ids.json
RESOURCES = REPO / "resources"

CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", str(DATA / "selected-contexts")))
TRAIN_JSON = DATA / "train_QA.json"
TEST_JSON = Path(os.environ.get("TEST_JSON", str(DATA / "private-official_QA.json")))

# ── segment ──────────────────────────────────────────────────────────────
MAX_CHUNK_WORDS = 250      # độ dài tối đa 1 chunk (từ)
SPLIT_STRIDE = 200         # bước cắt chunk con của Điều dài; chunk con giữ span Điều mẹ

# ── index / retrieve ─────────────────────────────────────────────────────
BM25_B = 1.0               # tham số b của BM25 (k1 = 1,2 đặt trong build_index.py)
DENSE_MODEL = str(MODELS / "Vietnamese_Embedding")   # AITeamVN/Vietnamese_Embedding
CE_MODEL = str(MODELS / "Vietnamese_Reranker")       # AITeamVN/Vietnamese_Reranker
ALPHA = float(os.environ.get("ALPHA", "0.85"))   # trọng số dense: (1-α)·BM25 + α·dense (env chỉ để ablation)
POOL = 200                 # số chunk giữ lại sau fusion
CE_DEPTH = 50              # số chunk đầu pool được cross-encoder chấm lại

# ── evidence ─────────────────────────────────────────────────────────────
TARGET_WORDS = 550         # độ dài tối đa đoạn trích (từ)
MIN_WORDS = 180            # widen lấy max(MIN_WORDS, TARGET_WORDS) từ; với giá trị hiện tại = 550
K4_FRAC = 0.85             # K4: ngưỡng hỗ trợ (tỷ lệ so điểm chunk cao nhất) để giữ một khoản
K4_MIN_KEEP = 250         # K4 không cắt nếu phần giữ lại < 250 từ
K4_PAD = 80               # K4 chỉ xét khi Điều dài hơn đoạn trích > 80 từ

# ── compose ──────────────────────────────────────────────────────────────
RESTATE_WORDS_PER_PART = 60
TRIM_DANGLE, TRIM_KEEP_QUOTE, TRIM_KEEP_TAIL = 45, 30, 20   # cắt câu cụt: tối đa 45 từ dư, giữ ≥30 (trích) / ≥20 (gạch đầu dòng)
MARKER = "Theo đó, nội dung nêu trên được quy định như sau:"
BOILER_TAIL = "(Hình từ Internet)\n..."   # dòng đuôi (~18% đáp án train có dòng này)

# ── memory (replay gold TRAIN, cos ≥ 0,95 + chốt thực thể) ───────────────
MEM_THRESHOLD = 0.95
ADMIN_TERMS = ["đô thị", "nông thôn", "cấp huyện", "cấp tỉnh", "cấp xã",
               "trung ương", "địa phương", "cấp trên", "cấp dưới"]
