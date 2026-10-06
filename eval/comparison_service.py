"""Isolated local comparison service for a frozen DepictQA adapter experiment.

This does not replace production weights. Generation uses the author service's
native settings and prompt; an optional adapter changes parameters only.
"""
import argparse
import json
from pathlib import Path
import sys


def validate_adapter(state, parameters):
    expected={k for k in parameters if 'lora_A' in k or 'lora_B' in k or k.startswith('vision_proj.')}
    if set(state)!=expected:
        raise ValueError('adapter must contain exactly the LoRA and visual projector parameters')
    for name,value in state.items():
        if value.shape != parameters[name].shape:
            raise ValueError('adapter shape mismatch: '+name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--identity',required=True)
    a=p.parse_args()
    import torch,yaml
    from easydict import EasyDict
    from flask import Flask,request
    from waitress import serve
    from .manifest_io import sha256
    from .experiment_support import atomic_json
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'DepictQA/src'))
    from model.depictqa import DepictQA
    cfgpath=root/'DepictQA/experiments/agenticir/config_comp.yaml'
    cfg=EasyDict(yaml.safe_load(cfgpath.read_text()))
    for k in ['vision_encoder_path','llm_path','delta_path']:
        cfg.model[k]=str((root/'DepictQA'/cfg.model[k]).resolve())
    model=DepictQA(cfg,training=False)
    status=model.load_state_dict(torch.load(cfg.model.delta_path,map_location='cpu'),strict=False)
    if status.unexpected_keys:raise ValueError('unexpected base delta keys')
    state=torch.load(a.adapter,map_location='cpu')
    params=dict(model.named_parameters());validate_adapter(state,params)
    with torch.no_grad():
        for name,value in state.items():params[name].copy_(value)
    model=model.eval().half().cuda()
    atomic_json(a.identity,{'adapter_sha256':sha256(a.adapter),'adapter_tensors':len(state),'author_delta_sha256':sha256(cfg.model.delta_path),'config_sha256':sha256(cfgpath),'infer':dict(cfg.infer),'host':'127.0.0.1','port':5002,'promotion':False,'generation':'native author implementation and settings; no response cache'})
    app=Flask(__name__)
    @app.route('/compare_quality',methods=['POST'])
    def query():
        if set(request.form)!={'imageA_path','imageB_path','prompt'}:
            return {'error':'invalid fields'},400
        images=[Path(request.form[k]) for k in ['imageA_path','imageB_path']]
        if not all(p.is_file() for p in images):return {'error':'missing image'},400
        with torch.no_grad():
            texts=model.generate({'query':[request.form['prompt']],'img_path':[None],
                'img_A_path':[str(images[0])],'img_B_path':[str(images[1])],
                'task_type':'quality_compare_noref',**{k:cfg.infer[k] for k in ['temperature','top_p','max_new_tokens','output_prob_id','output_confidence','sentence_model']}})[0]
        return {'answer':texts[0].strip()}
    serve(app,host='127.0.0.1',port=5002,threads=1)

if __name__=='__main__':main()
