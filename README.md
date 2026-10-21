# UIT - Kitchen — DSC 2026 Task 2 (LegalQA): mã nguồn tái lập kết quả Private Test

| | |
|---|---|
| Bài nộp | `submission.zip`, 1.918 câu, nộp 21/09/2026 07:54, ID 935934 |
| Kết quả Private | **Hạng 2** · **METEOR 0.6433** · ROUGE-L 0.511 |
| sha256 ZIP đã nộp | `dbb4a11b09552721d7e9f3590ad8263ffa60e77fac661a0016a7303f0588dcdc` (repo chỉ kèm sha256 từng đáp án ở `resources/reference_submission.sha256.json` để so sánh ở bước cuối) |
| sha256 `submission.json` bên trong | `1ece1f973e1a04a2dc4dc89a2e31150b27a50d1dc3d61df040fd2ca19209c14a` |
| Mô hình | 4 model zero-shot, **2.328.123.393 tham số** (58,2% trần 4 tỷ), đều trong danh mục BTC phê duyệt |

Tái lập **từ số 0**: `bash run_from_scratch.sh` tự dựng lại mọi index, embedding, pool ứng viên và điểm reranker từ dữ liệu BTC (`data/`) và 4 model gốc (`models/`). Cần GPU; khoảng 2 giờ 50 phút với 3 GPU (mục 4).

---

## 1. Hệ thống

Đáp án được **trích nguyên văn** điều luật trong corpus `selected-contexts`. Hệ thống **không có mô hình sinh văn bản**; bốn model chỉ làm truy hồi và xếp hạng.

```
câu hỏi ─┬─ [F] Nền v11 ──────────────────────────────────────────────────────────────────────────┐
         │   BM25 (bm25s) + dense Vietnamese_Embedding (0,15·BM25 + 0,85·dense) trên 424.722 chunk  │
         │   → Vietnamese_Reranker chấm 50 chunk đầu → chọn Điều (max-pool theo văn bản)            │
         │   → đoạn trích ≤ 550 từ, cắt đuôi K4 → khung đáp án (dẫn "Căn cứ…" / trích / "Như vậy…") │
         │   → memory: câu hỏi gần trùng câu TRAIN (cos ≥ 0,95 + khớp thực thể) dùng lại đáp án train│
         │   → tail: thêm câu hỏi đầu đáp án, túi từ khoá từ láng giềng train, sửa dòng "Căn cứ"     │
         │                                                                        private_v11.json │
         └─ [P] Pool ứng viên ────────────────────────────────────────────────┐                     │
             BM25 trên văn bản tách từ underthesea (Elasticsearch) top-100    │                     │
             ∪ Qwen3-Embedding-0.6B trên 192.670 Điều (MaxP) — RRF k = 60     │                     │
             → Vietnamese_Reranker chấm 15 văn bản × 3 chunk, sắp theo CE      │                     │
         [J] 39 ứng viên CẤP ĐIỀU/câu: 3 văn bản đầu × 8 chunk (bigram ∩ câu hỏi) ∪ chunk CE        │
             ∪ đoạn trích đang dùng → jina-reranker-v3.5 xếp hạng listwise (xuôi + ngược)           │
             → chèn tối đa 2 dòng evidence 160 từ ngay TRƯỚC đoạn trích nếu ứng viên khác đoạn    ◄─┘
               đang dùng (trùng 4-gram < 0,6) → ở câu private trùng 100% câu hỏi train/warmup (BTC
               không chấm) thay đáp án train bằng đoạn corpus → submission.zip
```

### Mô hình (mỗi model đếm 1 lần)

| thư mục `models/` | HuggingFace (danh mục BTC) | revision | tham số | vai trò |
|---|---|---|---:|---|
| `Vietnamese_Embedding` | AITeamVN/Vietnamese_Embedding | `dea33aa1ab339f38d66ae0a40e6c40e0a9249568` | 567.754.752 | dense nền v11, memory, láng giềng train |
| `Vietnamese_Reranker` | AITeamVN/Vietnamese_Reranker | `f536976248403314225d7fdfdbc87f0e9516a54e` | 567.755.777 | cross-encoder (nền v11 + pool Jina) |
| `Qwen3-Embedding-0.6B` | Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 595.776.512 | dense cấp Điều cho pool Jina |
| `jina-reranker-v3.5` | jinaai/jina-reranker-v3.5 | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` | 596.836.352 | xếp hạng listwise ứng viên evidence |
| **Tổng** | | | **2.328.123.393** | < 4.000.000.000 |

- **Không fine-tune, không LoRA, không quantize.** Cả 4 model chạy zero-shot với trọng số gốc.
- **Thư viện, không phải model:** `underthesea` (tách từ) và BM25 (`bm25s`, Elasticsearch). Cả hai cũng có trong danh mục BTC.
- **Kiểm revision:** các revision ở bảng trên đã được đối chiếu 29/09/2026. Mọi file model đội dùng trùng sha (LFS sha256 / git blob) với đúng revision đó trên HuggingFace.

### Tuân thủ thể lệ — `python3 check_compliance.py`

Script này đếm tham số từ header `.safetensors`, đối chiếu danh mục BTC, và grep mã nguồn cùng các script chạy để chắc không gọi API.

- **Trần tham số:** 2.328.123.393 < 4 tỷ.
- **Danh mục phê duyệt:** 4/4 model có trong danh mục BTC. `jina-reranker-v3.5` dùng license CC-BY-NC-4.0; thể lệ cho phép giấy phép phi thương mại.
- **Không API:** model nạp từ đĩa (`HF_HUB_OFFLINE=1`, `local_files_only=True`). Bước Jina chạy với `--network none`. Elasticsearch chạy cục bộ. Chỉ `download_models.sh` truy cập HuggingFace, và chỉ để tải trọng số.
- **Dữ liệu:**
  - Chỉ đọc `train_QA.json`, `selected-contexts/`, `private-official_QA.json`, tất cả từ `data/` của repo. Không dữ liệu ngoài, không augmentation.
  - `LegalQA_warmup.json` **chỉ** được `strip_banned_gold.py` dùng để *nhận diện* câu private trùng qid/câu hỏi với warmup hoặc train (những câu BTC không chấm). Ở các câu đó, nếu đáp án trùng một đáp án gold train/warmup thì bị thay bằng đoạn corpus. Không có đáp án warmup nào được chép vào bài nộp.

---

## 2. Cấu trúc repo

```
README.md                 tài liệu này
DSC2026_Task2_LegalQA_Data_Overview.docx.pdf   mô tả bài toán và dữ liệu của BTC
Scoring-Program-LegalQA.zip                    chương trình chấm của BTC (METEOR + ROUGE-L)
download_models.sh        tải 4 model đúng revision về models/ rồi kiểm sha256 + số tham số
run_from_scratch.sh       chạy toàn bộ pipeline từ số 0 (mục 4)
verify.py                 kiểm định dạng BTC và đếm câu khác so với bài đã nộp qua sha256 từng đáp án (chỉ cần Python 3)
compare.py                so sánh chi tiết 2 bản nộp bất kỳ: câu khác, METEOR/ROUGE-L tương đồng, điểm trên câu có gold công khai
check_compliance.py       kiểm trần tham số / danh mục model / không API (chỉ cần Python 3)
MANIFEST.sha256           sha256 của mọi file trong repo + dữ liệu BTC (data/) + 4 model (models/) đã dùng
env/
  Dockerfile.fusion       Python 3.8.10, torch 2.4.1+cu121, transformers 4.46.3, sentence-transformers 3.2.1, bm25s 0.3.11,
                          underthesea 6.8.4, scikit-learn 1.3.0 — môi trường đã chạy nền v11 và mọi bước CPU
  Dockerfile.gpu          Python 3.11.9, torch 2.4.1+cu121, transformers 4.53.2 — môi trường đã chạy Qwen3 / CE pool / Jina
  requirements-*.txt      phiên bản pin (đo bằng pip freeze trên máy đã sinh bài nộp)
  docker-compose.es.yml   Elasticsearch 7.17.0
  lib.sh                  hàm dùng chung cho run_from_scratch.sh
resources/
  harness_651.json        651 qid train không đưa vào bank K40 (tập giữ lại lúc nghiên cứu; tham số cố định của bài nộp)
  idf.json                bảng IDF (từ → idf) dùng chọn span evidence; tính một lần trên ứng viên corpus của 256 câu train,
                          idf = ln((N+1)/(df+1)) + 1, giữ cố định như một tham số của bài nộp
  reference_submission.sha256.json  sha256 ZIP / submission.json / từng đáp án của bài đã nộp (verify.py đọc)
src/fusion/               nền v11: corpus, segment, build_index, retrieve, evidence, compose, memory, predict, merge_shards,
                          encode_questions, build_trainq_emb, tail, fix_header, config
src/retrieval/            build_catalog (corpus/chunk/Điều), index_es (2 index ES), dense_qwen3, dense_qa, pool_wseg, fuse (RRF),
                          score_ce, ce_pool; baseline_common.py + legal_units.py (tách văn bản pháp luật)
src/jina/                 prepare (39 ứng viên), run_chunk (Jina), compose, resolver, strip_banned_gold, pack
data/                     KHÔNG kèm theo — đặt dữ liệu BTC vào đây (mục 3); mọi bước chỉ đọc dữ liệu từ đây
models/                   KHÔNG kèm theo — `bash download_models.sh` tải về (mục 3)
```

---

## 3. Chuẩn bị

### 3.1 Máy

- Linux x86_64, Docker ≥ 20.10 kèm plugin `docker compose` v2, NVIDIA Container Toolkit (`docker run --gpus`), `curl`, Python 3 (chỉ để chạy `verify.py` / `check_compliance.py`, dùng thư viện chuẩn).
- GPU NVIDIA ≥ 11 GB, driver hỗ trợ CUDA 12.1. Mỗi shard Jina cần ~6 GB VRAM trống. Cần thêm RAM ≥ 32 GB, đĩa trống ≥ 25 GB. CPU nhiều nhân giúp nhanh hơn: tách từ 347.396 chunk tốn ~1,3 giờ-CPU và chạy song song theo số nhân.

### 3.2 Dữ liệu BTC → `data/`

```
data/train_QA.json
data/private-official_QA.json
data/LegalQA_warmup.json
data/selected-contexts/context_*.json      (8.532 file)
```

`data/` và `models/` phải là thư mục thật trong repo, không phải symlink: container chỉ mount thư mục repo.

Kiểm dữ liệu trùng bản đội đã dùng:

```bash
grep '  data/' MANIFEST.sha256 | sha256sum -c --quiet && echo OK
```

### 3.3 Model → `models/`

```bash
bash download_models.sh
```

Script build image GPU nếu chưa có, rồi tải 4 model đúng revision ở mục 1 (~4,5 GB, ~2–3 phút). Sau đó nó kiểm sha256 theo `MANIFEST.sha256` và chạy `check_compliance.py`. Nếu mạng cần đăng nhập HuggingFace, đặt `HF_TOKEN=...` trước lệnh.

### 3.4 Image Docker

`run_from_scratch.sh` tự build image nếu chưa có. Muốn build trước:

```bash
docker build -t dsc26qa-fusion:v20 -f env/Dockerfile.fusion env
docker build -t dsc26qa-gpu:v20    -f env/Dockerfile.gpu    env
```

Biến môi trường dùng chung:

| biến | ý nghĩa | mặc định |
|---|---|---|
| `GPUS` | danh sách GPU, vd `"0 1 2"` | `0` |
| `WORK` | thư mục output | `work_scratch/` trong repo |
| `ES_PORT` | cổng host của Elasticsearch | `9211` |
| `ES_URL` | địa chỉ ES | `http://127.0.0.1:$ES_PORT` |
| `QJSON` | file câu hỏi trong `data/` (`public-official_QA.json` để chạy public; dùng `WORK` riêng) | `private-official_QA.json` |
| `ALPHA`, `USE_CE` | chỉ để ablation nền F: `ALPHA=0` chỉ BM25, `ALPHA=1` chỉ dense, `USE_CE=0` bỏ cross-encoder (mỗi cấu hình một `WORK` riêng) | `0.85`, `1` |

---

## 4. Chạy lại từ số 0

```bash
bash download_models.sh                    # một lần
GPUS="0 1 2" bash run_from_scratch.sh      # output: work_scratch/submission/submission.zip ; log: work_scratch/logs/
QJSON=public-official_QA.json WORK=work_public GPUS="0 1 2" bash run_from_scratch.sh   # tuỳ chọn: cùng hệ thống trên tập public
ES_PORT=9211 docker compose -f env/docker-compose.es.yml down -v   # tắt ES khi xong
```

Khi chạy public, bước cuối không so với bài nộp (verify chỉ dành cho private); chấm bằng scorer của BTC.

- Script tự bật Elasticsearch (`env/docker-compose.es.yml`) nếu `ES_URL` chưa có ES chạy. Nếu cổng 9211 đã bị chiếm, chạy với `ES_PORT=9311` (hoặc cổng trống khác) cho cả 2 lệnh.
- Nền F chạy song song với chuỗi R → P. F2–F5 chạy trên GPU thứ hai của `GPUS`, R/P không dùng GPU đó; riêng F4 chia 4 shard lên mọi GPU nên có thể chồng với R2/P2. Trên card 11 GB dùng chung, nếu OOM thì chạy lại lệnh để resume.
- Mỗi bước tự bỏ qua nếu output đã hoàn tất, nên khi lỗi (vd OOM vì GPU bị chiếm) chỉ cần chạy lại đúng lệnh trên.

Thời gian tham khảo trên 3 × RTX 2080 Ti 11 GB (dùng chung) và 40 nhân CPU:

| bước | việc | môi trường | thời gian |
|---|---|---|---|
| R0 | `build_catalog.py`: corpus.jsonl (8.512 văn bản, 347.396 chunk), chunks.jsonl, articles.jsonl (192.670 Điều) | CPU | 6 phút |
| R1 | `index_es.py`: ES `dsc26_legal_chunks_v1` + `dsc26_legal_chunks_wseg_h77` (tách từ underthesea) | CPU + ES | 8 phút |
| R2 | `dense_qwen3.py encode`: Qwen3-Embedding 192.670 Điều | GPU | 42 phút |
| P0–P1 | `pool_wseg.py` (BM25 tách từ top-100), `dense_qa.py` (Qwen3 top-100 văn bản + 60 Điều) | CPU + ES, GPU | 2 phút |
| P2 | `fuse.py` (RRF k=60) → `score_ce.py --top 15` 2 shard → `ce_pool.py` | GPU | 7 phút |
| F0–F1 | `segment.py` (424.722 chunk), `build_index.py --bm25` | CPU | 4 phút |
| F2 | `build_index.py --dense`: Vietnamese_Embedding fp16 | GPU | 64 phút |
| F3 | `build_trainq_emb.py`: vector 7.000 câu hỏi train | GPU | < 1 phút |
| F4 | `predict.py` 4 shard + `merge_shards.py` (~2,8 s/câu/GPU) | GPU | 53 phút |
| F5–F6 | `encode_questions.py`, `tail.py`, `fix_header.py` → `private_v11.json` | GPU + CPU | 4 phút |
| J0 | `prepare.py`: 39 ứng viên/câu (tối đa 43) | CPU | 2 phút |
| J1 | `run_chunk.py`: jina-reranker-v3.5 fp16, 3 shard | GPU | 37 phút |
| J2 | `compose.py` → `strip_banned_gold.py` → `pack.py` → `verify.py` | CPU | 1 phút |

Tổng khoảng 2 giờ 50 phút với 3 GPU. Với 1 GPU, ước khoảng 5–6 giờ.

### Sai số kỳ vọng

- **Bước CPU tất định:** `build_catalog`, `index_es`, `tail`, `fix_header`, `fuse`, `ce_pool`, `prepare`, `compose`, `strip`, `pack`. Riêng `corpus.jsonl` phải có sha256 `a5a7d8d0231ec2f629d5480529696602ea34e837871c90dfbe973a711d573c2f`.
- **Nguồn lệch duy nhất:** số học fp16 trên GPU (encode Vietnamese_Embedding / Qwen3, cross-encoder, Jina) khi khác card hoặc driver. Ở câu sát ngưỡng, thứ hạng hoặc láng giềng có thể đổi, chủ yếu làm khác vài từ ở dòng túi từ khoá cuối đáp án. Tỷ lệ câu khác ở mức vài phần trăm.
- **Kiểm tra:** `verify.py` (chạy tự động ở cuối) in số câu khác so với bản đã nộp; `compare.py` in chi tiết mức khác:

```bash
pip install nltk==3.9.1 rouge-score==0.1.2 && python3 -c "import nltk; nltk.download('wordnet')"
python3 compare.py work_scratch/submission/submission.zip <bản_nộp_khác>/submission.zip
```

---

## 5. Xử lý sự cố

- **`CUDA error: out of memory` / `AssertionError: <số>` ở log `jina_s*.log` / `ce_s*.log`**: GPU không đủ VRAM trống. Jina cần ≥ 4 GB trống lúc khởi động và ~6 GB lúc chạy. Chọn GPU khác qua `GPUS=...` rồi chạy lại; các bước đã xong được bỏ qua.
- **ES không lên / cổng bận**: đặt `ES_PORT` sang cổng trống. Xem `docker logs dsc26qa_es`.
- **Quyền ghi file**: container chạy bằng user mặc định của image (root). Trên đĩa cục bộ, file trong `work_*/` và `models/` sẽ thuộc root; dùng `sudo` để xoá nếu cần. Không thêm `--user` khi `work_*/` nằm trên NFS: một số NAS ánh xạ UID khiến container không ghi được.
- **Chỉ chạy qua Docker**: các bước dùng đường dẫn container cố định (`/repo`, `/work`, `/model`), nên repo không hỗ trợ chạy trực tiếp trên host.
