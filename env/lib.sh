# Hàm dùng cho run_from_scratch.sh (source, không chạy trực tiếp; cần đặt WORK trước khi source).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMG_FUSION=${IMG_FUSION:-dsc26qa-fusion:v20}
IMG_GPU=${IMG_GPU:-dsc26qa-gpu:v20}
export ES_PORT=${ES_PORT:-9211}                  # cổng host của ES (docker-compose.es.yml đọc biến này)
ES_URL=${ES_URL:-http://127.0.0.1:$ES_PORT}
read -r -a GPU_LIST <<< "${GPUS:-0}"          # vd GPUS="0 1 2"
QJSON=${QJSON:-private-official_QA.json}       # file câu hỏi trong data/ (public-official_QA.json để chạy public)
NQ=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1], encoding='utf-8'))))" "$REPO/data/$QJSON")
mkdir -p "$WORK" "$WORK/logs"; WORK=$(cd "$WORK" && pwd)

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
ok() { [ -s "$1" ]; }
full() { [ -s "$1" ] && [ "$(wc -l < "$1")" -ge "$2" ]; }   # file jsonl ghi dần: đủ n dòng mới coi là xong

images() {
  docker image inspect "$IMG_FUSION" >/dev/null 2>&1 || docker build -t "$IMG_FUSION" -f "$REPO/env/Dockerfile.fusion" "$REPO/env"
  docker image inspect "$IMG_GPU" >/dev/null 2>&1 || docker build -t "$IMG_GPU" -f "$REPO/env/Dockerfile.gpu" "$REPO/env"
}

# dk <image> <gpu|-> <workdir trong /repo> <cmd...> : chạy 1 bước trong container, repo mount /repo (chỉ đọc), WORK mount /work.
dk() {
  local img=$1 gpu=$2 wd=$3 gflag=(); shift 3
  [ "$gpu" != - ] && gflag=(--gpus "device=$gpu")
  docker run --rm --network host -e HOME=/tmp "${gflag[@]}" \
    -e DATA=/repo/data -e MODELS=/repo/models -e WORK=/work -e ES_URL="$ES_URL" -e TEST_JSON="/repo/data/$QJSON" \
    ${ALPHA:+-e ALPHA=$ALPHA} ${USE_CE:+-e USE_CE=$USE_CE} \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$REPO":/repo:ro -v "$WORK":/work -w "/repo/$wd" "$img" "$@"
}
cpu() { dk "$IMG_FUSION" - "$@"; }

# shards <n> <fn>: chạy fn 0..n-1, shard i trên GPU_LIST[i % số GPU]; các shard cùng GPU chạy tuần tự, khác GPU chạy song song.
shards() {
  local n=$1 fn=$2 k g pids=()
  for k in "${!GPU_LIST[@]}"; do
    g=${GPU_LIST[$k]}
    ( for ((i = k; i < n; i += ${#GPU_LIST[@]})); do "$fn" "$i" "$g" || exit 1; done ) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { log "shard lỗi ($fn)"; return 1; }; done
}

es_up() {
  curl -s "$ES_URL" >/dev/null 2>&1 || docker compose -f "$REPO/env/docker-compose.es.yml" up -d
  local i; for i in $(seq 120); do
    curl -s "$ES_URL/_cluster/health" | grep -qE '"status":"(green|yellow)"' && return 0; sleep 5
  done; log "ES không lên sau 10 phút ($ES_URL), xem docker logs dsc26qa_es"; return 1
}

# Jina (jina-reranker-v3.5) chấm 1 shard trong container cách ly mạng (--network none), chỉ mount /input /output /model /run.py.
# Cần ~6 GB VRAM trống; run_chunk.py dừng ngay nếu GPU còn < JINA_MIN_FREE_MIB=4000 MiB trống (chạy lại script để resume).
jina_shard() {
  local s=$1 g=$2 d=$WORK/jina/shard_$1
  ok "$d/output/complete.json" && return 0
  docker run --rm --gpus device=$g --network none -e HOME=/tmp -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e JINA_FRAC=.95 -e JINA_MIN_FREE_MIB=4000 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$d/input:/input:ro" -v "$d/output:/output" -v "$REPO/src/jina/run_chunk.py:/run.py:ro" -v "$REPO/models/jina-reranker-v3.5:/model:ro" \
    --entrypoint python "$IMG_GPU" -u /run.py > "$WORK/logs/jina_s$s.log" 2>&1
}

# Các bước CPU cuối: base_v11 = tail.py (-> private_v9.json) + fix_header.py (-> private_v11.json, nền v11);
# jina_inputs = ứng viên cho Jina; finish = compose -> strip gold câu cấm -> pack ZIP -> verify (sha256 từng đáp án của bài đã nộp).
base_v11() {
  ok "$WORK/fusion/private_v11.json" && return 0
  log "F6 tail + fix_header -> private_v11.json (nền v11)"
  cpu src/fusion python3 tail.py /work/fusion/private_pred.json /work/fusion/private_qvecs.npy /work/fusion > "$WORK/logs/tail.log" 2>&1
  cpu src/fusion python3 fix_header.py /work/fusion/private_v9.json /work/fusion/private_v11.json | tee "$WORK/logs/fix_header.log"
}
jina_inputs() {
  ok "$WORK/jina/inputs.json" && return 0
  log "J0 prepare: 39 ứng viên cấp Điều/câu (K=3 văn bản × M=8 chunk)"
  cpu src/jina python3 prepare.py /work/jina 3 8 3 | tee "$WORK/logs/prepare.log"
}
finish() {
  log "J2 compose (evidence 160 từ, trùng 4-gram < 0,6, tối đa 2 dòng) -> strip -> pack -> verify"
  mkdir -p "$WORK/submission"
  cpu src/jina python3 compose.py /work/jina /work/submission/predictions.json | tee "$WORK/logs/compose.log"
  cpu src/jina python3 strip_banned_gold.py /work/submission/predictions.json /work/submission/predictions_clean.json | tee "$WORK/logs/strip.log"
  cpu src/jina python3 pack.py /work/submission/predictions_clean.json /work/submission/submission.zip | tee "$WORK/logs/pack.log"
  [ "$QJSON" = private-official_QA.json ] || return 0   # verify chỉ so với bài nộp private
  python3 "$REPO/verify.py" "$WORK/submission/submission.zip" | tee "$WORK/logs/verify.log"
}
