"""Local-only staged VLM feedback. Model labels are not human judgements."""
import argparse
import json
from pathlib import Path
import time
import requests
from .manifest_io import read_csv,write_csv,sha256
from .experiment_support import atomic_json,failure_category,fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=['severity','compare'],default='severity');p.add_argument('--limit',type=int,default=1000);p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[r for r in read_csv(a.results) if r['status']=='ok'];session=requests.Session();session.trust_env=False
    from pipeline.prompts import depictqa_evaluate_degradation_prompt,depictqa_compare_prompt
    jobs=[]
    if a.mode=='severity':
        seen=set()
        for r in rows:
            for stage in ['input','output']:
                key=(r['sample_id'],'input' if stage=='input' else r['method'])
                if key in seen:continue
                seen.add(key);jobs.append({'sample_id':r['sample_id'],'method':key[1],'stage':stage,'path':r[stage+'_path'],'domain':r.get('domain',''),'role':r.get('role',''),'scale':r.get('scale','')})
        for j in jobs:j.update(payload={'imageA_path':j['path'],'prompt':depictqa_evaluate_degradation_prompt.format(degradation='rain')},url='http://127.0.0.1:5001/evaluate_degradation')
        weights=Path('DepictQA/weights/delta/degra_eval.pt')
    else:
        for r in rows:
            # Both orders reveal positional bias; each is a real separate request.
            for order in ['input_first','output_first']:
                paths=[r['input_path'],r['output_path']]
                if order=='output_first':paths.reverse()
                jobs.append({'sample_id':r['sample_id'],'method':r['method'],'stage':order,'domain':r.get('domain',''),'role':r.get('role',''),'scale':r.get('scale',''),
                    'payload':{'imageA_path':paths[0],'imageB_path':paths[1],'prompt':depictqa_compare_prompt},'url':'http://127.0.0.1:5002/compare_quality'})
        weights=Path('DepictQA/weights/delta/DQ495K_Abstractor.pt')
    if a.repeats < 1: raise ValueError('repeats must be positive')
    jobs=[dict(j,repetition=str(rep)) for rep in range(a.repeats) for j in jobs]
    task_config=Path('DepictQA/experiments/agenticir/config_'+('eval' if a.mode=='severity' else 'comp')+'.yaml')
    identity={'repeats':a.repeats,'task_config_sha256':sha256(task_config),'input_output_sha256':{r['sample_id']+'/'+r['method']:{key:sha256(r[key+'_path']) for key in ['input','output']} for r in rows},'results_sha256':sha256(a.results),'mode':a.mode,'delta_sha256':sha256(weights),'code_sha256':sha256(__file__)};run_hash=fingerprint(identity)
    cfg=out/'identity.json'
    if cfg.exists() and json.loads(cfg.read_text())['fingerprint']!=run_hash:raise ValueError('evaluation identity changed')
    atomic_json(cfg,{'identity':identity,'fingerprint':run_hash})
    records=read_csv(out/'labels.csv') if (out/'labels.csv').exists() else []
    for j in jobs[:a.limit]:
        old=next((r for r in records if all(r[k]==j[k] for k in ['sample_id','method','stage','repetition'])),None)
        if old and old['status']=='ok':continue
        row={k:v for k,v in j.items() if k not in ['payload','url','path']};row.update(model='local_DepictQA',reference_provided=False,status='failed',answer='',error='',failure_category='');start=time.monotonic()
        try:
            response=session.post(j['url'],data=j['payload'],timeout=(3,120));response.raise_for_status();answer=response.json()['answer'].strip()
            if a.mode=='severity':
                if answer not in ['very low','low','medium','high','very high']:raise ValueError('invalid response: '+answer)
            elif not(('A' in answer) ^ ('B' in answer)):raise ValueError('invalid response: '+answer)
            row.update(status='ok',answer=answer)
            if a.mode=='compare':
                choose_a='A' in answer;row['prefers_output']=choose_a==(j['stage']=='output_first')
        except Exception as exc:row.update(error=str(exc),failure_category=failure_category(exc))
        row['duration_seconds']=time.monotonic()-start;row['request_count']=1
        records=[r for r in records if not all(r[k]==j[k] for k in ['sample_id','method','stage','repetition'])]+[row];write_csv(out/'labels.csv',records)
        print(j['sample_id'],j['method'],j['stage'],row['status'],row['answer'],flush=True)
        if row['failure_category']=='service_unavailable':break
if __name__=='__main__':main()
