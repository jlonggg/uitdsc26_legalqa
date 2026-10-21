"""jina-reranker-v3.5 chấm listwise mọi ứng viên của một câu trong MỘT prompt, 2 lượt (thứ tự xuôi và đảo ngược), điểm = cosine
(doc_embed, query_embed), khoá theo chunk_id. Chạy trong container: /input (inputs.json + hash.json), /output, /model, /run.py.
Ngưỡng VRAM qua env JINA_MIN_FREE_MIB / JINA_FRAC; resume theo qid; fallback OOM (lần đầu đưa NGUYÊN VĂN text; chỉ khi OOM mới cắt mỗi ứng viên còn 128 rồi 64 từ)."""
import hashlib,importlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
I=Path('/input');O=Path('/output')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,ensure_ascii=False));temp.replace(p)
def main():
    assert sha(I/'inputs.json')==json.load(open(I/'hash.json'))['inputs.json']
    rows=json.load(open(I/'inputs.json'));torch.set_num_threads(2);torch.manual_seed(1212)
    free,total=torch.cuda.mem_get_info();assert free>int(os.environ.get('JINA_MIN_FREE_MIB','4000'))*2**20,free
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('JINA_FRAC','.5')))
    tok=AutoTokenizer.from_pretrained('/model',local_files_only=True,padding_side='left',trust_remote_code=False)
    m=AutoModel.from_pretrained('/model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    assert sum(p.numel() for p in m.parameters())<650000000
    fmt=importlib.import_module(m.__class__.__module__).format_docs_prompts_func
    expected=[1024 if x=='sliding_attention' else None for x in m.config.layer_types]
    assert [x.self_attn.sliding_window for x in m.model.layers]==expected
    start=time.time();results=json.load(open(O/'scores.json')) if (O/'scores.json').exists() else {}
    for k,r in enumerate(rows):
        if r['qid'] in results:continue
        out={}
        for cut in (None,128,64):   # None = nguyên văn text; chỉ cắt khi OOM
          try:
            out={}
            for tag,docs in [('forward',r['docs']),('reverse',r['docs'][::-1])]:
              texts=[d['text'] if cut is None else ' '.join(d['text'].split()[:cut]) for d in docs]
              assert all(t not in r['question'] and all(t not in d for d in texts) for t in m.special_tokens.values())
              ids=tok(fmt(r['question'],texts,special_tokens=m.special_tokens,no_thinking=True),truncation=False)['input_ids']
              assert len(ids)<tok.model_max_length
              enc=tok.pad({'input_ids':[ids]},padding=True,return_tensors='pt').to('cuda')
              assert int((enc['input_ids']==m.doc_embed_token_id).sum())==len(docs)
              with torch.inference_mode():
                  v=m(**enc);s=F.cosine_similarity(v.doc_embeds.float(),v.query_embeds.float().expand_as(v.doc_embeds),dim=-1).cpu().numpy()[0]
              assert s.shape==(len(docs),) and np.isfinite(s).all()
              out[tag]={d['chunk_id']:float(x) for d,x in zip(docs,s)};out[tag+'_tokens']=len(ids)
            out['cut']=cut;break
          except torch.OutOfMemoryError:
            torch.cuda.empty_cache();print('OOM',r['qid'],'cut',cut,flush=True)
        assert 'cut' in out,f"OOM ở mọi mức cắt, qid {r['qid']} (chạy lại trên GPU trống hơn)"
        results[r['qid']]=out;save(O/'scores.json',results)
        if (k+1)%16==0:print('DONE',k+1,'/',len(rows),'seconds',round(time.time()-start),'peak_mib',torch.cuda.max_memory_allocated()//2**20,flush=True)
    assert set(results)=={r['qid'] for r in rows}
    save(O/'complete.json',{'n':len(results),'seconds':time.time()-start,'peak_gpu_mib':torch.cuda.max_memory_allocated()//2**20,
                           'inputs_sha256':sha(I/'inputs.json'),'scores_sha256':sha(O/'scores.json'),'model_config_sha256':sha(Path('/model/config.json')),
                           'model_code_sha256':sha(Path('/model/modeling.py')),'script_sha256':sha(Path('/run.py'))})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
