import json,time,torch,sacrebleu
B=json.load(open("bench20.json",encoding="utf-8")); REFS=[b["ref"] for b in B]
from transformers import AutoProcessor,AutoModelForCausalLM
W="/tmp/gemma_src"; proc=AutoProcessor.from_pretrained(W); model=AutoModelForCausalLM.from_pretrained(W,dtype="auto")
P="\nТы профессиональный переводчик. Твоя задача перевести текст с английского на русский.\nВАЖНО чтобы в ответе был только перевод без каких-либо комментариев и уточнений.\n\nТекст для перевода:\n"
def g(src):
    t=proc.apply_chat_template([{"role":"user","content":P+src}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    i=proc(text=t,return_tensors="pt").to(model.device); il=i["input_ids"].shape[-1]
    with torch.no_grad(): o=model.generate(**i,max_new_tokens=1024)
    r=proc.decode(o[0][il:],skip_special_tokens=False)
    try: return proc.parse_response(r)["content"]
    except: return proc.decode(o[0][il:],skip_special_tokens=True)
t=time.time(); hyps=[g(b["src"]) for b in B]
bleu=sum(sacrebleu.sentence_bleu(h,[r]).score for h,r in zip(hyps,REFS))/len(hyps)
json.dump({"bleu":bleu,"hyps":hyps},open("gemma_baseline_bench20.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(f"GEMMA_BASELINE_BENCH20 BLEU={bleu:.2f} ({time.time()-t:.0f}s)")
