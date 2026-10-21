"""Vector câu hỏi test (Vietnamese_Embedding, fp16) cho memory/bag của tail.py. usage: encode_questions.py <out_dir>"""
import json,os,sys,numpy as np
from config import DENSE_MODEL, TEST_JSON
from sentence_transformers import SentenceTransformer
pub=json.load(open(TEST_JSON, encoding='utf-8')); ids=list(pub)
enc=SentenceTransformer(DENSE_MODEL,device='cuda'); enc.half()
V=enc.encode([pub[q]['question'] for q in ids],batch_size=64,normalize_embeddings=True,show_progress_bar=False)
O=sys.argv[1]; np.save(O+'/private_qvecs.npy',V.astype(np.float16)); json.dump(ids,open(O+'/private_qvecs_ids.json','w')); print(V.shape)
