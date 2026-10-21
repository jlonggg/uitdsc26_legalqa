#!/usr/bin/env python3
"""Qwen3-Embedding-0.6B zero-shot trên 192.670 đơn vị cấp Điều (articles.jsonl): passage = "header\nbody" cắt 320 token,
câu hỏi có tiền tố Instruct, cắt 192 token; vector = hidden state token cuối, chuẩn hoá L2, lưu fp16.
usage: dense_qwen3.py encode      -> WORK/retrieval/qwen3_zero.npy (fp16, cùng thứ tự dòng articles.jsonl)"""
import json, sys, time, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(os.environ.get("MODELS", f"{_REPO}/models"), "Qwen3-Embedding-0.6B")
O = os.path.join(os.environ.get("WORK", f"{_REPO}/work"), "retrieval")
ARTICLES = f"{O}/articles.jsonl"; VEC = f"{O}/qwen3_zero.npy"
PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
MAX_Q, MAX_D = 192, 320

def load():
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, local_files_only=True).cuda().eval()
    assert sum(p.numel() for p in model.parameters()) == 595_776_512
    return tok, model

@torch.inference_mode()
def encode(tok, model, texts, query, max_len):
    if query: texts = [PREFIX + t for t in texts]
    b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    b = {k: v.cuda() for k, v in b.items()}
    out = model(**b).last_hidden_state
    return F.normalize(out[:, -1].float(), dim=-1)

def article_text(r):
    head = r["header"] or ""
    return f"{head}\n{r['body']}" if head else r["body"]

def meta():
    return [json.loads(l) for l in open(ARTICLES, encoding="utf-8")]

def encode_corpus(bs=32):
    rows = meta(); tok, model = load()
    vec = np.lib.format.open_memmap(VEC + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(rows), 1024))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["body"]))  # length-sorted batches
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        v = encode(tok, model, [article_text(rows[i]) for i in idx], False, MAX_D).cpu().numpy().astype(np.float16)
        vec[idx] = v
        if (s // bs) % 200 == 0: print(f"{s}/{len(rows)} {(s+bs)/(time.time()-t0):.0f}/s", flush=True)
    vec.flush(); del vec
    os.replace(VEC + ".tmp.npy", VEC); print("done", len(rows))


if __name__ == "__main__" and sys.argv[1] == "encode":
    encode_corpus()
