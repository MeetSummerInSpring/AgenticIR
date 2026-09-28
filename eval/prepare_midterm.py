"""Read-only pack audit, fixed 512px-long-edge derivatives and dev previews."""
import argparse
import collections
import io
import json
import random
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .manifest_io import load_manifest, write_csv, sha256


def prepare(packs, out, split='dev', bases=4, real_limit=2):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.csv').exists(): raise FileExistsError('use a new output directory')
    audit=[]; all_rows=[]
    for pack in packs:
        pack=Path(pack).resolve(); rows=load_manifest(pack/'manifest.csv')
        checked=0; errors=[]
        for line in (pack/'checksums.sha256').read_text(encoding='utf-8-sig').splitlines():
            digest, name=line.split(maxsplit=1); path=(pack/name.lstrip('*')).resolve()
            if not path.is_relative_to(pack): raise ValueError('checksum path escapes pack')
            if not path.exists() or sha256(path)!=digest: errors.append(name)
            checked+=1
        for r in rows:
            with Image.open(r['input_path']) as im: im.verify()
        audit.append({'pack':str(pack),'hashes_checked':checked,'hash_failures':errors,
            'counts':dict(collections.Counter(r['role']+'/'+r['split'] for r in rows))})
        if errors: raise ValueError('pack checksum errors: '+str(errors))
        all_rows.extend(rows)
    (out/'pack_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    if len({r['sample_id'] for r in all_rows})!=len(all_rows): raise ValueError('cross-pack duplicate ID')
    write_csv(out/'source_manifest.csv',all_rows)
    selected=[]
    for domain in sorted({r['domain'] for r in all_rows}):
        candidates=[r for r in all_rows if r['domain']==domain and r['role']=='base' and r['split']==split]
        # Round-robin groups to avoid selecting only one source.
        groups=collections.defaultdict(list)
        for r in candidates: groups[r['group_id']].append(r)
        chosen=[]
        while len(chosen)<bases and any(groups.values()):
            for group in sorted(groups):
                if groups[group] and len(chosen)<bases: chosen.append(groups[group].pop(0))
        selected += chosen
        selected += [r for r in all_rows if r['domain']==domain and r['role']=='real' and r['split']==split][:real_limit]
    prepared=[]
    for source in selected:
        r=dict(source); im=Image.open(r['input_path']).convert('RGB'); original=im.size
        im.thumbnail((512,512),Image.Resampling.LANCZOS)
        path=out/'prepared'/f"{r['sample_id']}.png"; path.parent.mkdir(exist_ok=True); im.save(path)
        r.update(source_image_path=r['input_path'],input_path=str(path),image_path=str(path),
            original_size=json.dumps(original),output_size=json.dumps(im.size),adaptation='RGB; long_edge<=512 Lanczos; no crop/padding',input_sha256=sha256(path))
        prepared.append(r)
    write_csv(out/'prepared_manifest.csv',prepared)
    return prepared


def synthesize(rows,out,vendor,generators):
    from dataset.add_single_degradation import add_rain
    out=Path(out).resolve(); vendor=Path(vendor).resolve()
    # Author's AugMix code is retained with original copyright notices.
    import sys
    sys.path.insert(0,str(vendor)); import augment_and_mix
    z=zipfile.ZipFile(vendor/'Streaks_Garg06.zip')
    assets=sorted(n for n in z.namelist() if n.startswith('Streaks_Garg06/') and n.endswith('.png'))
    generated=[]; qa=[]; tiles=[]
    for idx,r in enumerate(q for q in rows if q['role']=='base'):
        base=Image.open(r['input_path']).convert('RGB'); x=np.asarray(base); row_tiles=[('reference '+r['domain'],base)]
        for gen in generators:
            for level in [1,2]:
                seed=2700+idx*10+level; random.seed(seed); np.random.seed(seed)
                params={}; asset=''
                if gen=='legacy_rain':
                    value={1:50,2:90}[level]; result=add_rain(x,value=value); params={'value':value}
                    replay=np.random.RandomState(seed); params['length']=int(replay.randint(20,40)); params['angle']=int(replay.randint(-30,30))
                else:
                    # Texture only is transformed. Geometry/color of the base stays fixed.
                    asset=assets[(idx*17+level)%len(assets)]
                    texture=Image.open(io.BytesIO(z.read(asset))).convert('RGB').resize(base.size,Image.Resampling.BILINEAR)
                    layer=augment_and_mix.augment_and_mix(np.asarray(texture,dtype=np.float32)/255.,severity=3,width=3,depth=-1,alpha=1.)
                    alpha={1:0.3,2:0.6}[level]; result=np.rint(np.clip(x/255.+alpha*layer,0,1)*255).astype(np.uint8)
                    params={'severity':3,'width':3,'depth':-1,'alpha_dirichlet':1.,'layer_gain':alpha,'composition':'clip(J+gain*AugMix(texture))'}
                sid=f"{r['sample_id']}__{gen}_{level}"; p=out/'synthetic'/f'{sid}.png'; p.parent.mkdir(exist_ok=True); Image.fromarray(result).save(p)
                q=dict(r); q.update(sample_id=sid,base_id=r['sample_id'],role='synthetic',input_path=str(p),image_path=str(p),reference_path=r['input_path'],
                    input_sha256=sha256(p),reference_sha256=sha256(r['input_path']),condition=f'{gen}/rain/L{level}',generator=gen,
                    generator_version=sha256(__file__),author_commit=(vendor/'source_commit.txt').read_text(),asset_id=asset,
                    asset_source='EfficientDeRain/rainmix/Streaks_Garg06.zip' if asset else '',asset_archive_sha256=sha256(vendor/'Streaks_Garg06.zip') if asset else '',
                    seed=seed,parameters=json.dumps(params),degradation_order='rain only')
                generated.append(q); row_tiles.append((gen+f' L{level}',Image.fromarray(result)))
                qa.append({'sample_id':sid,'saturated_fraction':float(np.mean(np.any(result==255,axis=2))),
                    'mean_absolute_change':float(np.mean(np.abs(result.astype(float)-x))), 'shape_preserved':result.shape==x.shape})
        tiles.append(row_tiles)
    for domain in sorted({r['domain'] for r in rows}):
        tiles.append([(domain+' real dev',Image.open(r['input_path'])) for r in rows if r['domain']==domain and r['role']=='real'])
    sheet=Image.new('RGB',(5*260,len(tiles)*175),'#202020'); draw=ImageDraw.Draw(sheet)
    for j,row in enumerate(tiles):
        for i,(label,im) in enumerate(row):
            im=im.copy(); im.thumbnail((256,145)); sheet.paste(im,(i*260,j*175+24)); draw.text((i*260+3,j*175+4),label,fill='white')
    sheet.save(out/'synthesis_preview.jpg',quality=92)
    write_csv(out/'synthesis_qa.csv',qa)
    # Include real dev; no base rows are scored as restored outputs.
    manifest=generated+[r for r in rows if r['role']=='real']
    write_csv(out/'manifest.csv',manifest)
    write_csv(out/'input_identity_results.csv',[{'sample_id':r['sample_id'],'method':'input_identity','input_path':r['input_path'],
        'output_path':r['input_path'],'reference_path':r['reference_path'],'status':'identity_control','error':''} for r in manifest])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--packs',nargs='+',default=['0927/street_pack_v1','0927/tianlian_pack_v1']); p.add_argument('--out',default='experiments/midterm_v1')
    p.add_argument('--vendor',default='experiments/midterm_v1/vendor'); p.add_argument('--split',choices=['dev','test'],default='dev'); p.add_argument('--bases-per-domain',type=int,default=4); p.add_argument('--real-per-domain',type=int,default=2); p.add_argument('--generators',nargs='+',choices=['legacy_rain','rainmix_adapted'],default=['legacy_rain','rainmix_adapted'])
    a=p.parse_args(); rows=prepare(a.packs,a.out,a.split,a.bases_per_domain,a.real_per_domain); synthesize(rows,a.out,a.vendor,a.generators)
if __name__=='__main__': main()
