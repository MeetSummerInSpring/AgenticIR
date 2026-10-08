"""Two native DepictQA routes sharing a backbone, with complete delta isolation.

The shared lock covers parameter replacement AND generation. Every request runs
vision and generation anew. CPU weight banks are not response/feature caches.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import time


class DeltaRouter:
    def __init__(self, model, banks):
        import torch
        self.model=model;self.lock=threading.RLock();self.active=None;self.switches=0
        self.targets=model.state_dict(keep_vars=True)
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        if not banks:raise ValueError('no delta branches')
        self.banks={}
        for mode,bank in banks.items():
            if set(bank)!=expected:raise ValueError('delta must cover exactly every trainable tensor: '+mode)
            for name,tensor in bank.items():
                if tensor.shape!=self.targets[name].shape:raise ValueError('delta shape mismatch: '+name)
                if not torch.isfinite(tensor).all():raise ValueError('nonfinite delta: '+name)
            self.banks[mode]={n:t.detach().to(device='cpu',dtype=self.targets[n].dtype).clone() for n,t in bank.items()}

    def generate(self, mode, inputs):
        import torch
        if mode not in self.banks:raise ValueError('unknown vision branch')
        with self.lock,torch.no_grad():
            if self.active!=mode:
                # Invalidate before replacement so a failed copy cannot claim readiness.
                self.active=None
                for name,value in self.banks[mode].items():self.targets[name].copy_(value)
                if any(t.is_cuda for t in self.targets.values()):torch.cuda.synchronize()
                self.active=mode;self.switches+=1
            return self.model.generate(inputs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    import torch,yaml
    from easydict import EasyDict
    from flask import Flask,request
    from waitress import create_server
    from .experiment_support import atomic_json
    from .manifest_io import sha256
    root=Path(__file__).resolve().parents[1];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'DepictQA/src'))
    from model.depictqa import DepictQA
    configs={mode:yaml.safe_load((root/'DepictQA/experiments/agenticir'/filename).read_text()) for mode,filename in [('severity','config_eval.yaml'),('compare','config_comp.yaml')]}
    normalized=[]
    for cfg in configs.values():
        c=copy.deepcopy(cfg);c['model'].pop('delta_path');normalized.append(c)
    if normalized[0]!=normalized[1]:raise ValueError('native branches do not share identical architecture/inference settings')
    banks={mode:torch.load(root/'DepictQA'/cfg['model']['delta_path'],map_location='cpu') for mode,cfg in configs.items()}
    cfg=EasyDict(copy.deepcopy(configs['severity']))
    for name in ['vision_encoder_path','llm_path','delta_path']:cfg.model[name]=str((root/'DepictQA'/cfg.model[name]).resolve())
    model=DepictQA(cfg,training=False).eval().half().cuda()
    router=DeltaRouter(model,banks);del banks
    identity={'deployment':'shared_backbone_full_delta_switch','branches':{mode:{'delta_sha256':sha256(root/'DepictQA'/c['model']['delta_path']),'delta_path':c['model']['delta_path'],'tensor_count':len(router.banks[mode])} for mode,c in configs.items()},'mutual_exclusion':'one lock across delta copy and generation','response_cache':False,'feature_cache':False,'host':'127.0.0.1','ports':[5001,5002],'infer':dict(cfg.infer)}
    atomic_json(out/'dual_identity.json',identity)
    def app_for(mode):
        app=Flask(mode);route='/evaluate_degradation' if mode=='severity' else '/compare_quality'
        @app.get('/health')
        def health():
            return {'ready':True,'branch':mode,'active_branch':router.active,'switches':router.switches,**identity['branches'][mode]}
        @app.post(route)
        def query():
            fields={'imageA_path','prompt'} if mode=='severity' else {'imageA_path','imageB_path','prompt'}
            if set(request.form)!=fields:return {'error':'invalid fields'},400
            if not all(Path(request.form[k]).is_file() for k in fields if k!='prompt'):return {'error':'missing image'},400
            started=time.monotonic()
            inputs={'query':[request.form['prompt']],'img_path':[None],'img_A_path':[request.form['imageA_path']],'img_B_path':[None if mode=='severity' else request.form['imageB_path']],'task_type':'quality_single_A_noref' if mode=='severity' else 'quality_compare_noref',**{k:cfg.infer[k] for k in ['temperature','top_p','max_new_tokens','output_prob_id','output_confidence','sentence_model']}}
            try:answer=router.generate(mode,inputs)[0][0].strip()
            except Exception:
                # Requests after CUDA failures must not blindly continue a poisoned model.
                raise
            with (out/'dual_requests.jsonl').open('a') as f:f.write(json.dumps({'branch':mode,'seconds':time.monotonic()-started,'switches':router.switches,'delta_sha256':identity['branches'][mode]['delta_sha256'],'answer':answer})+'\n')
            return {'answer':answer}
        return app
    servers=[create_server(app_for(mode),host='127.0.0.1',port=port,threads=1) for mode,port in [('severity',5001),('compare',5002)]]
    worker=threading.Thread(target=servers[0].run,daemon=True);worker.start();servers[1].run()

if __name__=='__main__':main()
