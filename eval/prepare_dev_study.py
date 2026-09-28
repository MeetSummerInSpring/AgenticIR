"""Dev-only synthesis comparison and paired 512/1024 diagnostics."""
import argparse
import io
import json
import math
from pathlib import Path
import random
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from .manifest_io import load_manifest,read_csv,write_csv,sha256
from .experiment_support import atomic_json




def composite(x,layer,target,mode):
    # Match an input-only perturbation statistic, never a restoration metric.
    def apply(gain):
        return np.clip(x+gain*layer*(1-x if mode=='screen' else 1),0,1)
    low,high=0.,1.
    for _ in range(24):
        mid=(low+high)/2
        if float(np.mean(apply(mid)-x))<target:low=mid
        else:high=mid
    gain=(low+high)/2
    return apply(gain),gain


def sheet(rows,path,width=300,height=190):
    columns=max(map(len,rows));canvas=Image.new('RGB',(columns*width,len(rows)*height),'#222222');draw=ImageDraw.Draw(canvas)
    for j,row in enumerate(rows):
        for i,(label,im) in enumerate(row):
            im=im.copy();im.thumbnail((width-4,height-25));canvas.paste(im,(i*width,j*height+23));draw.text((i*width+3,j*height+3),label,fill='white')
    canvas.save(path,quality=93)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='experiments/midterm_v2/data');p.add_argument('--selection',default='experiments/midterm_v2/data/selection.json');a=p.parse_args();out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'candidate_manifest.csv').exists():raise FileExistsError('do not overwrite frozen derived inputs')
    from dataset.add_single_degradation import add_rain
    real_ids=json.loads(Path(a.selection).read_text())['real']
    source=[]
    for domain in real_ids:source+=load_manifest('0927/'+domain+'_pack_v1/manifest.csv')
    index={r['sample_id']:r for r in source}
    old=read_csv('experiments/midterm_v1/prepared_manifest.csv');bases=[index[r['sample_id']] for r in old if r['role']=='base']
    reals=[index[sid] for ids in real_ids.values() for sid in ids]
    assert all(r['split']=='dev' for r in bases+reals)
    z=zipfile.ZipFile('experiments/midterm_v1/vendor/Streaks_Garg06.zip');assets=sorted(n for n in z.namelist() if n.startswith('Streaks_Garg06/') and n.endswith('.png'))
    archive_hash=sha256('experiments/midterm_v1/vendor/Streaks_Garg06.zip');author_commit=Path('experiments/midterm_v1/vendor/source_commit.txt').read_text().strip()
    candidates=[];prepared=[];qa=[];visual=[]
    for i,base in enumerate(bases):
        original=Image.open(base['input_path']).convert('RGB');im=original.copy();im.thumbnail((1024,1024),Image.Resampling.LANCZOS)
        ref=out/'references'/f"{base['sample_id']}.png";ref.parent.mkdir(exist_ok=True);im.save(ref);x=np.asarray(im,dtype=np.float32)/255
        seed=9280+i;target=.020 if i%2==0 else .035;np.random.seed(seed);random.seed(seed)
        oldlayer=add_rain(np.zeros_like(np.asarray(im)),value=70).astype(np.float32)/255
        asset=assets[(i*17+1)%len(assets)];texture=Image.open(io.BytesIO(z.read(asset))).convert('L')
        # Preserve texture aspect ratio; randomized texture crop, never crop J.
        scale=max(im.width/texture.width,im.height/texture.height)*1.12
        texture=texture.resize((math.ceil(texture.width*scale),math.ceil(texture.height*scale)),Image.Resampling.BILINEAR)
        rng=np.random.RandomState(seed);ox=rng.randint(texture.width-im.width+1);oy=rng.randint(texture.height-im.height+1)
        texture=texture.crop((ox,oy,ox+im.width,oy+im.height));layer=np.asarray(texture,dtype=np.float32)/255
        normalization=float(np.quantile(layer,.995));layer=np.clip(layer/max(normalization,1e-6),0,1)[...,None]
        tiles=[(base['domain']+' / J',im)]
        for name,l,mode in [('legacy_add_matched',oldlayer,'add'),('texture_add_matched',layer,'add'),('texture_screen_matched',layer,'screen')]:
            y,gain=composite(x,l,target,mode);result=Image.fromarray(np.rint(y*255).astype('uint8'))
            sid=base['sample_id']+'__'+name;pout=out/'candidates'/f'{sid}.png';pout.parent.mkdir(exist_ok=True);result.save(pout)
            row=dict(base);row.update(sample_id=sid,base_id=base['sample_id'],role='synthetic',input_path=str(pout),image_path=str(pout),reference_path=str(ref),source_image_path=base['input_path'],
                condition=name+('/L1' if i%2==0 else '/L2'),seed=seed,generator=name,scale=1024,input_sha256=sha256(pout),reference_sha256=sha256(ref),
                asset_id=asset if name.startswith('texture') else '',asset_archive_sha256=archive_hash,asset_author_commit=author_commit,
                generator_sha256=sha256(__file__),parameters=json.dumps({'target_mean_added_intensity':target,'gain':gain,'texture_scale':scale,'texture_crop':[int(ox),int(oy),im.width,im.height],'texture_p995':normalization,'composition':mode}),
                processing_order='RGB resize J to long-edge 1024, generate rain at 1024; no further crop',original_size=json.dumps(original.size),output_size=json.dumps(im.size))
            candidates.append(row);tiles.append((name,result));qa.append({'sample_id':sid,'target_mean_change':target,'actual_mean_change':float(np.mean(np.asarray(result)/255-x)),
                'new_saturated_fraction':float(np.mean(np.any((np.asarray(result)==255)&(np.asarray(im)<255),axis=2))), 'gain':gain})
        visual.append(tiles)
    for r in reals:
        im=Image.open(r['input_path']).convert('RGB');im.thumbnail((1024,1024),Image.Resampling.LANCZOS);path=out/'real'/f"{r['sample_id']}.png";path.parent.mkdir(exist_ok=True);im.save(path)
        q=dict(r);q.update(source_image_path=r['input_path'],input_path=str(path),image_path=str(path),input_sha256=sha256(path),scale=1024,condition='real_observed',processing_order='RGB resize long-edge1024; frozen road crop retained',output_size=json.dumps(im.size));prepared.append(q)
    for domain in real_ids:
        visual.append([(domain+' real dev '+str(j),Image.open(r['input_path'])) for j,r in enumerate(prepared) if r['domain']==domain])
    sheet(visual,out/'synthesis_comparison.jpg')
    write_csv(out/'candidate_manifest.csv',candidates);write_csv(out/'real_manifest.csv',prepared);write_csv(out/'synthesis_checks.csv',qa)
    # Resolution diagnostics use identical 1024 degradation and downsample both I/J to 512.
    chosen=[candidates[2],candidates[14],prepared[0],prepared[4]];resolution=[];panels=[]
    for r in chosen:
        row=[]
        for size in [512,1024]:
            q=dict(r);q['sample_id']=r['sample_id']+'__s'+str(size);q['scale']=size;q['condition']=r['condition']+'/scale'+str(size)
            for field in ['input','reference']:
                if not r[field+'_path']:continue
                im=Image.open(r[field+'_path']);im.thumbnail((size,size),Image.Resampling.LANCZOS);dest=out/'resolution'/f"{q['sample_id']}__{field}.png";dest.parent.mkdir(exist_ok=True);im.save(dest);q[field+'_path']=str(dest);q[field+'_sha256']=sha256(dest)
            resolution.append(q);im=Image.open(q['input_path']);cx,cy=im.width//2,im.height//2;patch=im.crop((max(0,cx-120),max(0,cy-100),min(im.width,cx+120),min(im.height,cy+100)))
            row.append((r['domain']+' '+r['role']+' '+str(size)+' native crop',patch))
        panels.append(row)
    write_csv(out/'resolution_manifest.csv',resolution);sheet(panels,out/'resolution_native_crops.jpg',300,240)
    atomic_json(out/'selection.json',{'bases':[r['sample_id'] for r in bases],'real':real_ids,'dev_only':True,'candidate_count':len(candidates),'restoration_scores_used_for_selection':False})
if __name__=='__main__':main()
