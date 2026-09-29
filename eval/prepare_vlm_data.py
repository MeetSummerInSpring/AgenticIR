"""Auditable grouped content-preservation supervision; no VLM pseudo-labels."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .manifest_io import sha256,write_csv
from .experiment_support import atomic_json

PROMPT=('Compare visible content preservation and image quality for the same scene. '
        'Answer exactly A, B, Tie, or Uncertain. Prefer intact visible detail over clipping, occlusion, '
        'compression artifacts or added noise. Use Tie for indistinguishable images. '
        'If the scenes do not correspond, or a preference cannot be established, answer Uncertain.')


def prepare(pack,protocol,out):
    pack=Path(pack).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('choose a new data directory')
    cfg=json.loads(Path(protocol).read_text());rows=list(csv.DictReader(open(pack/'manifest.csv',encoding='utf-8-sig')))
    selected=[r for r in rows if r['split']=='dev' and r['group_id'] in cfg['train_groups']+cfg['val_groups']]
    if any(r['domain']!='street' for r in selected):raise ValueError('street-only training/selection')
    groups={k:set(cfg[k+'_groups']) for k in ['train','val']}
    if groups['train']&groups['val'] or (groups['train']|groups['val'])&set(cfg['test_groups_excluded']):raise ValueError('group leakage')
    out.mkdir(parents=True);sources=[];pairs=[];items=[]
    for i,r in enumerate(sorted(selected,key=lambda x:x['sample_id'])):
        split='train' if r['group_id'] in groups['train'] else 'val';sid=r['sample_id'];folder=out/'images'/sid;folder.mkdir(parents=True)
        original=Image.open(pack/r['image_path']).convert('RGB');original.thumbnail((1024,1024),Image.Resampling.LANCZOS);original.save(folder/'original.png')
        x=np.asarray(original,dtype=np.float32)/255.;rng=np.random.RandomState(929000+i);variants={}
        variants['clipping']=np.clip(x*1.9+.12,0,1)
        variants['noise']=np.clip(x+rng.normal(0,.10,x.shape),0,1)
        blocked=x.copy();h,w=x.shape[:2];blocked[int(.28*h):int(.65*h),int(.28*w):int(.60*w)]=.98;variants['occlusion']=blocked
        for name,y in variants.items():Image.fromarray(np.rint(y*255).astype('uint8')).save(folder/(name+'.png'))
        for quality in [8,20]:
            buffer=io.BytesIO();original.save(buffer,format='JPEG',quality=quality);buffer.seek(0);Image.open(buffer).convert('RGB').save(folder/('jpeg'+str(quality)+'.png'))
        # Multi-direction translucent streaks are a controlled overlay, not a real-rain physical truth.
        rain=x.copy();mask=np.zeros((h,w),dtype=np.float32)
        for _ in range(max(200,w)):
            xx=rng.randint(w);yy=rng.randint(h);length=rng.randint(12,36)
            for dy in range(length):
                y0=yy+dy;x0=xx+dy//5
                if y0<h and x0<w:mask[y0,x0]=.45
        rain=np.clip(x+mask[...,None]*(1-x),0,1);Image.fromarray(np.rint(rain*255).astype('uint8')).save(folder/'rain_overlay.png')
        source=dict(sample_id=sid,group_id=r['group_id'],scene_id=r['scene_id'],split=split,original_role=r['role'],lighting=r['lighting'],visual_tags=r['visual_tags'],source_path=str(pack/r['image_path']),source_sha256=sha256(pack/r['image_path']),input_path=str(folder/'original.png'),input_sha256=sha256(folder/'original.png'))
        sources.append(source)
        for name in ['clipping','noise','occlusion','jpeg8','jpeg20','rain_overlay']:
            pairs.append(dict(pair_id=sid+'__'+name,sample_id=sid,group_id=r['group_id'],split=split,left=str(folder/'original.png'),right=str(folder/(name+'.png')),canonical_label='left',kind='controlled_'+name,basis='Known added impairment; content-preservation objective, not human MOS or a guaranteed ranking of restoration tools',label_origin='construction',human_reviewed=False))
        # Same image remains a real-world unchanged negative control for unnecessary processing.
        pairs.append(dict(pair_id=sid+'__identity',sample_id=sid,group_id=r['group_id'],split=split,left=str(folder/'original.png'),right=str(folder/'original.png'),canonical_label='Tie',kind='real_identity',basis='Exact pixel identity',label_origin='exact_identity',human_reviewed=False))
    for split in ['train','val']:
        eligible=[r for r in sources if r['split']==split]
        for i,r in enumerate(eligible):
            others=[q for q in eligible if q['group_id']!=r['group_id'] and q['scene_id']!=r['scene_id']]
            if not others:continue
            other=others[i%len(others)]
            pairs.append(dict(pair_id=r['sample_id']+'__noncorresponding',sample_id=r['sample_id'],group_id=r['group_id'],other_group_id=other['group_id'],split=split,left=r['input_path'],right=other['input_path'],canonical_label='Uncertain',kind='real_noncorrespondence',basis='Different scene identifiers and groups; prompt explicitly requires abstaining on noncorrespondence, not a global quality assertion',label_origin='source_metadata',human_reviewed=False))
    for p in pairs:
        for order in ['forward','reverse']:
            a,b=(p['left'],p['right']) if order=='forward' else (p['right'],p['left'])
            label=('A' if order=='forward' else 'B') if p['canonical_label']=='left' else p['canonical_label']
            items.append(dict(p,id=p['pair_id']+'__'+order,order=order,image_A=a,image_B=b,image_A_sha256=sha256(a),image_B_sha256=sha256(b),prompt=PROMPT,answer=label))
    for split in ['train','val']:
        subset=[q for q in items if q['split']==split];atomic_json(out/(split+'.json'),subset)
    atomic_json(out/'pairs.json',pairs);write_csv(out/'sources.csv',sources)
    report={'independent_source_images':len(sources),'groups':{k:sorted(v) for k,v in groups.items()},'sources_per_split':{s:sum(x['split']==s for x in sources) for s in groups},'pairs_per_split':{s:sum(x['split']==s for x in pairs) for s in groups},'qa_per_split':{s:sum(x['split']==s for x in items) for s in groups},'derived_files':sum(1 for p in (out/'images').rglob('*.png')),'prompt':PROMPT,'label_limitations':['No human real restoration preferences available','Controlled overlays are not a real raindrop-removal benchmark','Uncertain means metadata-proven noncorrespondence, not calibrated subtle-quality uncertainty','Development supervision alone does not establish final heldout or real restoration quality'],'protocol_sha256':sha256(protocol),'pack_manifest_sha256':sha256(pack/'manifest.csv')}
    atomic_json(out/'data_report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(a.pack,a.protocol,a.out)
