"""Real metric self-check; unavailable neural backbones are explicit skips."""
import argparse,json,math
from pathlib import Path
import numpy as np
from .score_manifest import Metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--weights-dir');a=p.parse_args()
    m=Metrics('cpu',a.weights_dir); x=np.random.RandomState(72).rand(192,192,3).astype('float32'); report={}
    for name in ['psnr','ssim','lpips','dists','niqe']:
        try:
            if name=='niqe':
                value=m.score(name,x); assert math.isfinite(value); report[name]={'status':'passed','value':value};continue
            identical=m.score(name,x,x); changed=m.score(name,x*.8,x)
            if name=='psnr': assert math.isinf(identical) and math.isfinite(changed)
            elif name=='ssim': assert abs(identical-1)<1e-5 and changed<identical
            else: assert abs(identical)<1e-5 and changed>identical
            report[name]={'status':'passed','identical':str(identical),'changed':changed}
        except (FileNotFoundError,RuntimeError,ImportError) as exc: report[name]={'status':'unavailable','reason':str(exc)}
        except Exception as exc: report[name]={'status':'failed','reason':str(exc)}
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if any(v['status']=='failed' for v in report.values()): raise SystemExit(1)
if __name__=='__main__':main()
