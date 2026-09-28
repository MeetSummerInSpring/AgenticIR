"""Pinned author UDR-S2Former: RGB [0,1], 320 tiles, 64 overlap, uniform merge.

No resizing, sharpening, downloaded weights or test-time augmentation. Smaller
inputs receive reflection padding to 320 and are unpadded after inference.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def restore(model,array):
    h,w=array.shape[:2]
    x=torch.from_numpy(array.astype(np.float32)/255).permute(2,0,1)[None].cuda()
    # Pad iteratively so reflection remains valid even for narrow images.
    while min(x.shape[-2:])<320:
        ph=min(max(320-x.shape[-2],0),max(x.shape[-2]-1,1))
        pw=min(max(320-x.shape[-1],0),max(x.shape[-1]-1,1))
        x=F.pad(x,(0,pw,0,ph),'reflect' if min(x.shape[-2:])>1 else 'replicate')
    H,W=x.shape[-2:];acc=torch.zeros_like(x);count=torch.zeros_like(x)
    with torch.inference_mode():
        for top in list(range(0,H-320,256))+[H-320]:
            for left in list(range(0,W-320,256))+[W-320]:
                pred=model(x[...,top:top+320,left:left+320])[0][0]
                acc[...,top:top+320,left:left+320]+=pred
                count[...,top:top+320,left:left+320]+=1
        y=(acc/count)[...,:h,:w].clamp(0,1)
    if not torch.isfinite(y).all():raise ValueError('nonfinite UDR output')
    return np.rint(y[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)


def main():
    p=argparse.ArgumentParser()
    for name in ['input_dir','output_dir','weights']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();paths=list(a.input_dir.iterdir())
    if len(paths)!=1 or any(a.output_dir.iterdir()):raise ValueError('one input and empty output directory required')
    vendor=Path(__file__).resolve().parent/'tools/UDR-S2Former';sys.path.insert(0,str(vendor))
    import UDR_S2Former as author
    torch.manual_seed(928);torch.cuda.manual_seed_all(928);torch.backends.cudnn.benchmark=False
    author.device=torch.device('cuda:0')
    model=author.Transformer(img_size=(320,320))
    model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True),strict=True)
    model=model.eval().cuda()
    Image.fromarray(restore(model,np.asarray(Image.open(paths[0]).convert('RGB')))).save(a.output_dir/'output.png')


if __name__=='__main__':main()
