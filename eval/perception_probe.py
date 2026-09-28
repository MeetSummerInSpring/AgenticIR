"""Development-only local structure-sensitivity probe; not a recovery benchmark."""
import argparse
import json
from pathlib import Path
import time
import requests
from .manifest_io import read_csv,write_csv,sha256
from .experiment_support import atomic_json
from pipeline.prompts import depictqa_compare_prompt

STRUCTURE_PROMPT=('Which image better preserves clearly visible edges, small structures and local details, '
                  'with less local blur? Do not infer hidden content. Answer Image A or Image B.')


def prepare(manifest,out,cases_path):
    from PIL import Image,ImageFilter
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('immutable protocol already exists')
    index={r['sample_id']:r for r in read_csv(manifest)}
    specs=json.loads(Path(cases_path).read_text())
    cases=[]
    for spec in specs:
        sid=spec['sample_id'];domain=spec['domain'];relative=spec['roi']
        path=spec.get('path') or index[sid]['input_path']
        original=Image.open(path).convert('RGB');original.thumbnail((1024,1024),Image.Resampling.LANCZOS)
        w,h=original.size;box=tuple(round(v*(w if i%2==0 else h)) for i,v in enumerate(relative))
        damaged=original.copy();damaged.paste(original.crop(box).filter(ImageFilter.GaussianBlur(5)),box)
        folder=out/sid;folder.mkdir()
        for label,im in [('original',original),('damaged',damaged)]:
            im.save(folder/(label+'.png'));im.crop(box).save(folder/(label+'_roi.png'))
        cases.append(dict(sample_id=sid,domain=domain,split='dev',source_path=str(Path(path).resolve()),source_sha256=sha256(path),roi=box,folder=str(folder),files={p.name:sha256(p) for p in folder.glob('*.png')}))
    protocol={'purpose':'detect known local loss of visible high-frequency signal; no clean-reference/MOS/real recovery claim','transformation':'Gaussian blur radius5 inside prespecified ROI; outside unchanged','prompts':{'quality':depictqa_compare_prompt,'structure':STRUCTURE_PROMPT},'cases':cases,'design':'damaged full+ROI x2 prompts x2 orders; identical full x2 prompts x2 orders; 48 requests','selection':'existing business dev and repository public natural examples; no holdout; no training'}
    atomic_json(out/'protocol.json',protocol)


def run(out):
    out=Path(out);protocol=json.loads((out/'protocol.json').read_text());session=requests.Session();session.trust_env=False
    result=out/'results.csv';rows=read_csv(result) if result.exists() else [];done={r['key'] for r in rows if r['status']=='ok'}
    for case in protocol['cases']:
        folder=Path(case['folder'])
        for name,digest in case['files'].items():
            if sha256(folder/name)!=digest:raise ValueError('probe image changed')
        for variant,view in [('damage','full'),('damage','roi'),('identity','full')]:
            suffix='_roi' if view=='roi' else ''
            original=folder/('original'+suffix+'.png');candidate=folder/(('damaged' if variant=='damage' else 'original')+suffix+'.png')
            for prompt_name,prompt in protocol['prompts'].items():
                for order in ['original_first','original_second']:
                    key='|'.join([case['sample_id'],variant,view,prompt_name,order])
                    if key in done:continue
                    a,b=(original,candidate) if order=='original_first' else (candidate,original)
                    row=dict(key=key,sample_id=case['sample_id'],domain=case['domain'],split='dev',variant=variant,view=view,prompt=prompt_name,order=order,status='failed',answer='',choice='',original_selected='',error='');started=time.monotonic()
                    try:
                        response=session.post('http://127.0.0.1:5002/compare_quality',data={'imageA_path':str(a),'imageB_path':str(b),'prompt':prompt},timeout=120);response.raise_for_status();answer=response.json()['answer'];row['answer']=answer
                        choice='A' if 'A' in answer and 'B' not in answer else 'B' if 'B' in answer and 'A' not in answer else ''
                        if not choice:raise ValueError('unparsed choice')
                        row.update(status='ok',choice=choice,original_selected=int(choice==('A' if order=='original_first' else 'B')) if variant=='damage' else '')
                    except Exception as exc:row['error']=str(exc)
                    row['seconds']=round(time.monotonic()-started,3);rows=[r for r in rows if r['key']!=key]+[row];write_csv(result,rows);print(key,row['status'],row['choice'],flush=True)
                    if row['status']!='ok':raise RuntimeError('stop after probe failure; retain evidence')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--manifest');p.add_argument('--cases');p.add_argument('--out',required=True);a=p.parse_args()
    if a.action=='prepare':prepare(a.manifest,a.out,a.cases)
    else:run(a.out)
if __name__=='__main__':main()
