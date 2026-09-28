"""One dev refinement prompted by local qualitative review, before metric access."""
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from .manifest_io import read_csv,write_csv,sha256
from .prepare_dev_study import composite,sheet
from .experiment_support import atomic_json


def main():
    root=Path('experiments/midterm_v2/data');out=root/'fine_texture';out.mkdir(exist_ok=True)
    if (out/'manifest.csv').exists():raise FileExistsError('refinement is immutable')
    prior=[r for r in read_csv(root/'candidate_manifest.csv') if r['generator']=='texture_screen_matched'];new=[];visual=[]
    z=zipfile.ZipFile('experiments/midterm_v1/vendor/Streaks_Garg06.zip')
    for i,r in enumerate(prior):
        ref=Image.open(r['reference_path']);x=np.asarray(ref,dtype=np.float32)/255.;asset=Image.open(io.BytesIO(z.read(r['asset_id']))).convert('L')
        # Keep source texture at 0.6x native pixels instead of >2x stretching.
        asset=asset.resize((round(asset.width*.6),round(asset.height*.6)),Image.Resampling.BILINEAR)
        rng=np.random.RandomState(int(r['seed'])+100);base=np.asarray(asset,dtype=np.float32)/255.;height,width=x.shape[:2];layer=np.zeros((height,width),dtype=np.float32)
        # Overlap independent shifted tiles with Hann weights to suppress seams.
        ah,aw=base.shape;stepy=max(1,ah//2);stepx=max(1,aw//2);weights=np.zeros_like(layer);window=np.maximum(np.outer(np.hanning(ah),np.hanning(aw)),1e-5)
        for yy in range(-ah,height,stepy):
            for xx in range(-aw,width,stepx):
                tile=np.roll(base,(rng.randint(ah),rng.randint(aw)),axis=(0,1));y0=max(0,yy);x0=max(0,xx);y1=min(height,yy+ah);x1=min(width,xx+aw)
                if y1<=y0 or x1<=x0:continue
                sy=slice(y0-yy,y1-yy);sx=slice(x0-xx,x1-xx);layer[y0:y1,x0:x1]+=tile[sy,sx]*window[sy,sx];weights[y0:y1,x0:x1]+=window[sy,sx]
        layer/=np.maximum(weights,1e-8);p995=float(np.quantile(layer,.995));layer=np.clip(layer/max(p995,1e-6),0,1)[...,None]
        target=json.loads(r['parameters'])['target_mean_added_intensity'];y,gain=composite(x,layer,target,'screen');im=Image.fromarray(np.rint(y*255).astype('uint8'));sid=r['base_id']+'__texture_screen_fine';path=out/(sid+'.png');im.save(path)
        q=dict(r);q.update(sample_id=sid,input_path=str(path.resolve()),image_path=str(path.resolve()),input_sha256=sha256(path),generator='texture_screen_fine',condition='texture_screen_fine/'+r['condition'].split('/')[-1],
            generator_sha256=sha256(__file__),seed=int(r['seed'])+100,parameters=json.dumps({'asset_native_scale':.6,'tile_overlap':.5,'weight':'Hann','target_mean_added_intensity':target,'actual_mean_added_intensity':float(np.mean(np.asarray(im)/255-x)),'gain':gain,'normalization_p995':p995,'composition':'screen'}))
        new.append(q);visual.append([(r['domain']+' reference',ref),('coarse screen prior',Image.open(r['input_path'])),('fine screen dev revision',im)])
    write_csv(out/'manifest.csv',new);sheet(visual,out/'fine_comparison.jpg')
    atomic_json(out/'decision_basis.json',{'cause':'street qualitative review says texture streaks too thick; reduce native texture scale, preserve target perturbation','restoration_metrics_used':False,'test_used':False,'new_dev_inputs':8,'prior_outputs_preserved':True})
if __name__=='__main__':main()
