"""Supplement free-format metrics with shared option likelihood and native prompt."""
import argparse,json
from pathlib import Path
import torch
from .vlm_training import load_model,set_adapters,batch,evaluate_rows,save_json,digest


def score_options(model,rows,out,tag):
    model.eval();records=[]
    for i,row in enumerate(rows):
        scores={}
        for label in ['A','B','Tie','Uncertain']:
            candidate=dict(row,answer=label)
            with torch.no_grad(),torch.cuda.amp.autocast():scores[label]=-float(model(batch(candidate))[0])
        winner=max(scores,key=scores.get)
        records.append({k:row.get(k,'') for k in ['id','pair_id','sample_id','group_id','kind','split','answer','order','domain','label_origin','human_reviewed']}|{'scores':scores,'prediction':winner,'correct':winner==row['answer'] if row['answer'] in ['A','B','Tie','Uncertain'] else None})
        if (i+1)%32==0:print(tag,'choices',i+1,'/',len(rows),flush=True)
    labels={label:[r for r in records if r['answer']==label] for label in ['A','B','Tie','Uncertain']}
    class_acc={label:sum(r['correct'] for r in rs)/len(rs) for label,rs in labels.items() if rs}
    labeled=[r for r in records if r['correct'] is not None]
    summary={'n':len(records),'n_labeled':len(labeled),'accuracy':sum(r['correct'] for r in labeled)/len(labeled) if labeled else None,'macro_class_accuracy':sum(class_acc.values())/len(class_acc) if class_acc else None,'class_accuracy':class_acc,'method':'negative mean teacher-forced response token loss, including author response separator; shared choice set; no free-format parsing','no_reselection':True}
    save_json(Path(out)/(tag+'_choice_predictions.json'),records);save_json(Path(out)/(tag+'_choice_summary.json'),summary);print(tag,summary,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--adapter',required=True);p.add_argument('--cache',required=True);p.add_argument('--native',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=json.loads(Path(a.data).read_text());model,identity=load_model(a.cache)
    from pipeline.prompts import depictqa_compare_prompt
    native=[dict(r,prompt=depictqa_compare_prompt) for r in rows if r['answer'] in ['A','B']]
    for tag in ['baseline','trained']:
        if tag=='trained':set_adapters(model,torch.load(a.adapter,map_location='cpu'))
        score_options(model,rows,out,tag)
        if a.native:evaluate_rows(model,native,out,tag+'_native_prompt')
    save_json(out/'identity.json',{'data_sha256':digest(a.data),'adapter_sha256':digest(a.adapter),'source_sha256':digest(__file__),'model_identity':identity})
if __name__=='__main__':main()
