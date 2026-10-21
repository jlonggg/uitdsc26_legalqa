"""Retrieval của nền v11:
    · điểm chunk = (1-α)·BM25/max + α·minmax(cosine dense) trên TOÀN BỘ chunk, α = 0,85; lấy pool 200 chunk
    · cross-encoder Vietnamese_Reranker chấm lại 50 chunk đầu (cặp câu hỏi, tiêu đề + chunk), sắp theo điểm CE
    · văn bản = max-pool điểm chunk; trả về văn bản/chunk thắng, pool (cho K4) và margin top1−top2"""
from __future__ import annotations

import numpy as np

from config import ALPHA, ART, CE_DEPTH, CE_MODEL, DENSE_MODEL, POOL
from segment import load_store


class Retriever:
    def __init__(self, device: str = "cuda", use_ce: bool = True):
        self.st = load_store()
        import bm25s
        self.bm = bm25s.BM25.load(str(ART / "bm25s_index"))
        self._bm25s = bm25s
        self.emb = np.load(ART / "emb_aiteam.npy", mmap_mode="r")
        self.emb_t = None                      # FUSION_GPU_DENSE=1: tính dense trên GPU; bản nộp KHÔNG bật (CPU fp32)
        if __import__("os").environ.get("FUSION_GPU_DENSE") and device == "cuda":
            import torch
            self.emb_t = torch.from_numpy(np.ascontiguousarray(self.emb)).to(device)
        from sentence_transformers import SentenceTransformer
        self.enc = SentenceTransformer(DENSE_MODEL, device=device)
        if device == "cuda":
            self.enc.half()
        self.ce = None
        if use_ce:
            from sentence_transformers import CrossEncoder
            import torch
            self.ce = CrossEncoder(CE_MODEL, device=device, max_length=512,
                                   automodel_args={"torch_dtype": torch.float16})

    @staticmethod
    def _dense_all(emb, qv: np.ndarray) -> np.ndarray:
        """cosine với TẤT CẢ chunk (fp32 trên CPU), không giới hạn vào ứng viên BM25."""
        n = emb.shape[0]
        out = np.empty(n, dtype=np.float32)
        q = qv.astype(np.float32)
        B = 100_000
        for s in range(0, n, B):
            out[s:s + B] = emb[s:s + B].astype(np.float32) @ q
        return out

    def _dense_fused(self, question: str) -> np.ndarray:
        """minmax(cosine Vietnamese_Embedding) trên toàn corpus."""
        def mm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-9)
        qa = self.enc.encode([question], normalize_embeddings=True)[0]
        if self.emb_t is not None:
            import torch
            with torch.no_grad():
                d = (self.emb_t @ torch.from_numpy(qa.astype(np.float16)).to(self.emb_t.device)).float().cpu().numpy()
            d = mm(d)
        else:
            d = mm(self._dense_all(self.emb, qa))
        return d

    def search_chunks(self, question: str, topk: int = POOL):
        """-> [(ci, score)]: 50 chunk đầu pool fusion, sắp theo điểm cross-encoder (cặp: câu hỏi, tiêu đề + "\\n" + chunk)."""
        qtok = self._bm25s.tokenize([question], stopwords=None,
                                    show_progress=False)
        k_bm = min(2000, self.emb.shape[0])
        ids, sc = self.bm.retrieve(qtok, k=k_bm, show_progress=False)
        ids, sc = ids[0], sc[0].astype(np.float32)
        s_bm = np.zeros(self.emb.shape[0], dtype=np.float32)
        s_bm[ids] = sc                              # ngoài top-2000 ≈ min = 0
        s_bm /= (s_bm.max() + 1e-9)
        d = self._dense_fused(question)
        fused = (1 - ALPHA) * s_bm + ALPHA * d
        k = min(topk * 4, len(fused))
        cand = np.argpartition(-fused, k - 1)[:k]
        cand = cand[np.argsort(-fused[cand])]
        pool = [(int(i), float(fused[i])) for i in cand[:topk]]
        if self.ce is not None:
            head = pool[:CE_DEPTH]
            pairs = [(question,
                      self.st.titles[self.st.doc_of_chunk[ci]] + "\n"
                      + self.st.texts[ci]) for ci, _ in head]
            ces = np.asarray(self.ce.predict(pairs, batch_size=16,
                                             show_progress_bar=False),
                             dtype=np.float32)
            order = np.argsort(-ces)
            pool = [(head[j][0], float(ces[j])) for j in order]
        return pool

    def pick(self, question: str, topk: int = POOL):
        """-> (doc_idx, ci_thắng, pool, doc_chunks, margin)"""
        pool = self.search_chunks(question, topk)
        best: dict[int, tuple[float, int]] = {}
        for ci, sc in pool:
            di = self.st.doc_of_chunk[ci]
            if di not in best or sc > best[di][0]:
                best[di] = (sc, ci)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        di, (s1, ci) = ranked[0]
        margin = s1 - ranked[1][1][0] if len(ranked) > 1 else 99.0
        doc_chunks = [(self.st.texts[c], s) for c, s in pool
                      if self.st.doc_of_chunk[c] == di]
        return di, ci, pool, doc_chunks, float(margin)
