#!/usr/bin/env bash
# Tái lập bài nộp TỪ SỐ 0: chỉ dùng dữ liệu BTC (data/) + 4 model gốc HuggingFace (models/, tải bằng download_models.sh).
# Mọi index / embedding / pool / điểm Jina tự sinh lại trong $WORK; bước cuối so với sha256 từng đáp án của bài đã nộp (resources/).
# Bốn nhóm bước: F (nền v11, chạy nền song song) · R (corpus dẫn xuất + ES + Qwen3) · P (pool ứng viên cho Jina) · J (Jina -> bản nộp).
#
# usage: GPUS="0 1 2" WORK=work_scratch bash run_from_scratch.sh
#   QJSON=public-official_QA.json WORK=work_public  chạy trên tập public (tên file trung gian vẫn là *private*; dùng WORK riêng)
#   ablation nền F (mỗi cấu hình một WORK riêng): ALPHA=0 (chỉ BM25) | ALPHA=1 (chỉ dense) | USE_CE=0 (bỏ cross-encoder)
#   Mỗi bước bỏ qua nếu output đã có (chạy lại để resume). Log: $WORK/logs/. Kết quả: $WORK/submission/submission.zip
#   Thời gian đo 29/09/2026 (3 × RTX 2080 Ti dùng chung, 40 nhân CPU): ~2 giờ 50 phút. Chi tiết từng bước trong README.md.
set -euo pipefail
WORK=${WORK:-$(cd "$(dirname "$0")" && pwd)/work_scratch}
source "$(dirname "$0")/env/lib.sh"
mkdir -p "$WORK"/{fusion/index,fusion/build,retrieval,jina}
G0=${GPU_LIST[0]}; G1=${GPU_LIST[1]:-$G0}   # R/P chạy trên G0, nền F (bước dài nhất) trên G1, hai chuỗi chạy song song
images

fusion_chain() {
# ── F. Nền v11 = đáp án dựng từ đoạn luật trích nguyên văn, không dùng mô hình sinh ──
#    retrieval Vietnamese_Embedding + BM25 + Vietnamese_Reranker -> compose -> tail.py (private_v9.json) -> fix_header.py (private_v11.json)
ok "$WORK/fusion/index/chunks.pkl" || { log "F0 segment"; cpu src/fusion python3 segment.py > "$WORK/logs/segment.log" 2>&1; }
ok "$WORK/fusion/index/bm25s_index/vocab.index.json" || { log "F1 BM25"; cpu src/fusion python3 build_index.py --bm25 > "$WORK/logs/bm25.log" 2>&1; }
ok "$WORK/fusion/index/emb_aiteam.npy" || { log "F2 dense Vietnamese_Embedding 424.722 chunk (GPU, ~1 giờ)"; dk "$IMG_FUSION" "$G1" src/fusion python3 build_index.py --dense > "$WORK/logs/dense.log" 2>&1; }
ok "$WORK/fusion/build/trainq_emb.npy" || { log "F3 bank câu hỏi train"; dk "$IMG_FUSION" "$G1" src/fusion python3 build_trainq_emb.py > "$WORK/logs/trainq.log" 2>&1; }
predict_shard() { ok "$WORK/fusion/pred_shard$1.json" || dk "$IMG_FUSION" "$2" src/fusion python3 predict.py --shard "$1/4" \
  --out "/work/fusion/pred_shard$1.json" --margins-out "/work/fusion/margins_shard$1.json" > "$WORK/logs/predict_s$1.log" 2>&1; }
if ! ok "$WORK/fusion/private_pred.json"; then
  log "F4 predict $NQ câu, 4 shard (GPU; memory replay gold TRAIN cos≥0,95)"; shards 4 predict_shard
  cpu src/fusion python3 merge_shards.py /work/fusion 4
fi
ok "$WORK/fusion/private_qvecs.npy" || { log "F5 vector câu hỏi"; dk "$IMG_FUSION" "$G1" src/fusion python3 encode_questions.py /work/fusion > "$WORK/logs/encode_q.log" 2>&1; }
base_v11

}
pool_chain() {
local GPU_LIST=("${GPU_LIST[@]}"); [ ${#GPU_LIST[@]} -gt 1 ] && GPU_LIST=("${GPU_LIST[@]/$G1}") && GPU_LIST=(${GPU_LIST[@]})   # bỏ GPU G1 của nền F để không OOM
# ── P. Pool ứng viên cho Jina: BM25-wseg ∪ Qwen3 dense (RRF 60) -> CE Vietnamese_Reranker top-15 doc ──
full "$WORK/retrieval/pool_private.jsonl" "$NQ" || { log "P0 BM25 wseg top-100 doc"; cpu src/retrieval python3 pool_wseg.py "/repo/data/$QJSON" /work/retrieval/pool_private.jsonl > "$WORK/logs/pool_wseg.log" 2>&1; }
full "$WORK/retrieval/dense_private.jsonl" "$NQ" || { log "P1 Qwen3 dense (GPU)"; dk "$IMG_GPU" "$G0" src/retrieval python3 dense_qa.py "/repo/data/$QJSON" /work/retrieval/dense_private.jsonl > "$WORK/logs/dense_qa.log" 2>&1; }
full "$WORK/retrieval/pool_private_rrf.jsonl" "$NQ" || cpu src/retrieval python3 fuse.py /work/retrieval/pool_private.jsonl /work/retrieval/dense_private.jsonl /work/retrieval/pool_private_rrf.jsonl /work/retrieval/chunks.jsonl
ce_shard() { full "$WORK/retrieval/ce_private_s$1.jsonl" $(((NQ + 1 - $1) / 2)) || \
  dk "$IMG_GPU" "$2" src/retrieval python3 score_ce.py /repo/models/Vietnamese_Reranker "/repo/data/$QJSON" \
  /work/retrieval/pool_private_rrf.jsonl "/work/retrieval/ce_private_s$1.jsonl" --top 15 --max-len 1024 --batch 16 --shard "$1" --nshards 2 > "$WORK/logs/ce_s$1.log" 2>&1; }
if ! full "$WORK/retrieval/pool_private_ce.jsonl" "$NQ"; then
  log "P2 cross-encoder top-15 doc × 3 chunk, 2 shard (GPU)"; shards 2 ce_shard
  cat "$WORK/retrieval/ce_private_s0.jsonl" "$WORK/retrieval/ce_private_s1.jsonl" > "$WORK/retrieval/ce_private.jsonl"
  cpu src/retrieval python3 ce_pool.py /work/retrieval/ce_private.jsonl /work/retrieval/pool_private_rrf.jsonl /work/retrieval/pool_private_ce.jsonl
fi

}
fusion_chain > "$WORK/logs/fusion_chain.log" 2>&1 & FPID=$!   # nền F chạy song song với R + P

# ── R. Kho corpus dẫn xuất + Elasticsearch ────────────────────────────────
full "$WORK/retrieval/articles.jsonl" 192670 || { log "R0 build_catalog (corpus.jsonl / chunks.jsonl / articles.jsonl)"; cpu src/retrieval python3 build_catalog.py /work/retrieval | tee "$WORK/logs/catalog.log"; }
es_up
log "R1 index ES dsc26_legal_chunks_v1 + dsc26_legal_chunks_wseg_h77 (underthesea, CPU)"
cpu src/retrieval python3 index_es.py /work/retrieval/corpus.jsonl > "$WORK/logs/index_es.log" 2>&1
ok "$WORK/retrieval/qwen3_zero.npy" || { log "R2 Qwen3-Embedding-0.6B encode 192.670 Điều (GPU)"; dk "$IMG_GPU" "$G0" src/retrieval python3 dense_qwen3.py encode > "$WORK/logs/qwen3_encode.log" 2>&1; }

pool_chain
wait $FPID || { log "nền F lỗi, xem $WORK/logs/fusion_chain.log"; exit 1; }

# ── J. Jina cấp Điều -> bản nộp ─────────────────────────────────────────
jina_inputs
log "J1 jina-reranker-v3.5 listwise 2 chiều, 3 shard (GPU, cần ~6 GB trống/shard)"; shards 3 jina_shard
finish
log "XONG -> $WORK/submission/submission.zip"
