"""Materialize a pre-frozen heldout protocol; never relabel test as development."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from .manifest_io import load_manifest, read_csv, write_csv, sha256
from .experiment_support import atomic_json
from .prepare_dev_study import composite


def fine_screen(reference, asset, seed, target):
    """Exact v2 fine-texture algorithm, generalized to a declared input/seed."""
    x=np.asarray(reference,dtype=np.float32)/255.
    asset=asset.convert('L');asset=asset.resize((round(asset.width*.6),round(asset.height*.6)),Image.Resampling.BILINEAR)
    rng=np.random.RandomState(seed);base=np.asarray(asset,dtype=np.float32)/255.;height,width=x.shape[:2]
    layer=np.zeros((height,width),dtype=np.float32);weights=np.zeros_like(layer);ah,aw=base.shape
    window=np.maximum(np.outer(np.hanning(ah),np.hanning(aw)),1e-5)
    for yy in range(-ah,height,max(1,ah//2)):
        for xx in range(-aw,width,max(1,aw//2)):
            tile=np.roll(base,(rng.randint(ah),rng.randint(aw)),axis=(0,1));y0=max(0,yy);x0=max(0,xx);y1=min(height,yy+ah);x1=min(width,xx+aw)
            if y1<=y0 or x1<=x0:continue
            sy=slice(y0-yy,y1-yy);sx=slice(x0-xx,x1-xx);layer[y0:y1,x0:x1]+=tile[sy,sx]*window[sy,sx];weights[y0:y1,x0:x1]+=window[sy,sx]
    layer/=np.maximum(weights,1e-8);p995=float(np.quantile(layer,.995));layer=np.clip(layer/max(p995,1e-6),0,1)[...,None]
    y,gain=composite(x,layer,target,'screen');result=Image.fromarray(np.rint(y*255).astype('uint8'))
    return result,{'gain':gain,'normalization_p995':p995,'actual_mean_added_intensity':float(np.mean(np.asarray(result)/255-x))}


def validate_test_execution(manifest, tools, plan_path):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text());protocol=Path(plan['frozen_protocol_path'])
    if sha256(protocol)!=plan['frozen_protocol_sha256']:raise ValueError('frozen protocol changed')
    identity=sha256(manifest)
    specification=next((x for x in plan['manifests'] if x['sha256']==identity),None)
    if specification is None or sorted(tools)!=sorted(specification['tools']):raise ValueError('test manifest/tools differ from frozen execution plan')
    rows=load_manifest(manifest)
    if len(rows)!=specification['n'] or any(r['split']!='test' for r in rows):raise ValueError('invalid test scope')
    for r in rows:
        if sha256(r['input_path'])!=r['input_sha256']:raise ValueError('test input changed')
        if r.get('reference_path') and sha256(r['reference_path'])!=r['reference_sha256']:raise ValueError('test reference changed')
    return plan['frozen_protocol_sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--out',required=True);p.add_argument('--packs',default='0927');a=p.parse_args()
    protocol=Path(a.protocol).resolve();cfg=json.loads(protocol.read_text());out=Path(a.out).resolve()
    if out.exists():raise FileExistsError('holdout derivatives must use a new directory')
    if cfg['split']!='test' or cfg['expected_input_count']!=104:raise ValueError('unsupported frozen protocol')
    if sha256('eval/refine_rain_texture.py')!=cfg['generator_source_sha256']:raise ValueError('generator baseline changed')
    archive=Path('experiments/midterm_v1/vendor/Streaks_Garg06.zip')
    if sha256(archive)!=cfg['asset_archive_sha256']:raise ValueError('asset changed')
    z=zipfile.ZipFile(archive);assets=sorted(n for n in z.namelist() if n.startswith('Streaks_Garg06/') and n.endswith('.png'))
    # Before touching test images, require exact output agreement on existing dev derivatives.
    dev=read_csv('experiments/midterm_v2/data/fine_texture/manifest.csv')
    for r in dev:
        target=json.loads(r['parameters'])['target_mean_added_intensity'];im,_=fine_screen(Image.open(r['reference_path']),Image.open(io.BytesIO(z.read(r['asset_id']))),int(r['seed']),target)
        if not np.array_equal(np.asarray(im),np.asarray(Image.open(r['input_path']))):raise ValueError('frozen v2 generator reproduction mismatch')
    out.mkdir(parents=True);real=[];synthetic=[]
    for di,domain in enumerate(cfg['domains']):
        source=load_manifest(Path(a.packs)/(domain+'_pack_v1')/'manifest.csv')
        for role,count in [('real',cfg['real_per_domain']),('base',cfg['base_per_domain'])]:
            rows=sorted([r for r in source if r['split']=='test' and r['role']==role],key=lambda r:r['sample_id'])
            if len(rows)!=count:raise ValueError('frozen pack count mismatch: '+domain+'/'+role)
            for i,r in enumerate(rows):
                image=Image.open(r['input_path']).convert('RGB');image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
                q={k:r.get(k,'') for k in ['sample_id','domain','role','split','group_id','visual_tags','lighting','normal_or_mild_control']}
                q.update(scale=1024,source_image_path=r['input_path'],reference_path='',base_id='',source_input_sha256=sha256(r['input_path']))
                if role=='real':
                    dest=out/'real'/(r['sample_id']+'.png');dest.parent.mkdir(exist_ok=True);image.save(dest);q.update(input_path=str(dest),input_sha256=sha256(dest),condition='real_observed');real.append(q)
                else:
                    ref=out/'references'/(r['sample_id']+'.png');ref.parent.mkdir(exist_ok=True);image.save(ref)
                    asset=assets[(17*i+1)%len(assets)];seed=929000+1000*di+i;target=.020 if i%2==0 else .035
                    result,params=fine_screen(image,Image.open(io.BytesIO(z.read(asset))),seed,target)
                    sid=r['sample_id']+'__texture_screen_fine';dest=out/'synthetic'/(sid+'.png');dest.parent.mkdir(exist_ok=True);result.save(dest)
                    q.update(sample_id=sid,base_id=r['sample_id'],role='synthetic',input_path=str(dest),input_sha256=sha256(dest),reference_path=str(ref),reference_sha256=sha256(ref),condition='texture_screen_fine/'+('L1' if i%2==0 else 'L2'),seed=seed,asset_id=asset,parameters=json.dumps(dict(params,target_mean_added_intensity=target)),generator='texture_screen_fine');synthetic.append(q)
    write_csv(out/'real_manifest.csv',real);write_csv(out/'synthetic_manifest.csv',synthetic);write_csv(out/'manifest.csv',real+synthetic)
    plan={'frozen_protocol_path':str(protocol),'frozen_protocol_sha256':sha256(protocol),'generator_dev_exact_matches':len(dev),'excluded_bases':[],'manifests':[{'path':str(out/name),'sha256':sha256(out/name),'n':len(rows),'tools':tools} for name,rows,tools in [('real_manifest.csv',real,cfg['real_tools']),('synthetic_manifest.csv',synthetic,cfg['synthetic_tools'])]],'preparation_source_sha256':sha256(__file__)}
    atomic_json(out/'execution_plan.json',plan);print('heldout prepared',len(real),'real',len(synthetic),'synthetic; exact dev reproduction',len(dev))
if __name__=='__main__':main()
