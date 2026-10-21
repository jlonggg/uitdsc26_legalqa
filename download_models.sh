#!/usr/bin/env bash
# Tải 4 model gốc từ HuggingFace về models/ đúng revision đã dùng cho bài nộp, rồi kiểm sha256 + số tham số.
# Chạy trong image GPU (có sẵn huggingface-cli), host không cần cài gì thêm. Cần Internet; ~4,5 GB.
# usage: bash download_models.sh        (HF_TOKEN=... nếu mạng yêu cầu đăng nhập HuggingFace)
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
IMG_GPU=${IMG_GPU:-dsc26qa-gpu:v20}
docker image inspect "$IMG_GPU" >/dev/null 2>&1 || docker build -t "$IMG_GPU" -f "$REPO/env/Dockerfile.gpu" "$REPO/env"
mkdir -p "$REPO/models"
while read -r dir repo rev; do
  echo "== $repo @ $rev -> models/$dir"
  docker run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} -v "$REPO/models":/models "$IMG_GPU" \
    huggingface-cli download "$repo" --revision "$rev" --local-dir "/models/$dir" --exclude "onnx/*" </dev/null
  docker run --rm -v "$REPO/models":/models "$IMG_GPU" rm -rf "/models/$dir/.cache" </dev/null   # metadata tải về, không thuộc model
done <<'EOF'
Vietnamese_Embedding AITeamVN/Vietnamese_Embedding dea33aa1ab339f38d66ae0a40e6c40e0a9249568
Vietnamese_Reranker AITeamVN/Vietnamese_Reranker f536976248403314225d7fdfdbc87f0e9516a54e
Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-0.6B 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
jina-reranker-v3.5 jinaai/jina-reranker-v3.5 e8a93f33f0b22108f8c2364f8484ce3422552fbc
EOF
cd "$REPO" && grep '  models/' MANIFEST.sha256 | sha256sum -c --quiet && echo "sha256: 4 model khớp bản đã dùng"
python3 "$REPO/check_compliance.py"
