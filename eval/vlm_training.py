"""Isolated single-GPU DepictQA adapter training; vendor and production weights unchanged."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def align_single_conversation(tokenize):
    """Match native batch-one generation: EOS inside the prompt is not padding."""
    def aligned(conversations, task_type):
        if len(conversations) != 1:
            raise ValueError('aligned attention supports exactly one unpadded conversation')
        ids, targets, mask = tokenize(conversations, task_type)
        # pad_sequence cannot add padding to a batch containing one sequence.
        return ids, targets, mask.new_ones(mask.shape)
    return aligned


def load_model(cache_dir,train_projector=True):
    import torch,yaml
    from easydict import EasyDict
    sys.path.insert(0,str(ROOT/'DepictQA/src'))
    from model.depictqa import DepictQA
    from model.model_llama import LlamaForCausalLM
    cfg=yaml.safe_load((ROOT/'DepictQA/experiments/agenticir/config_comp.yaml').read_text())
    for k in ['vision_encoder_path','llm_path','delta_path']:cfg['model'][k]=str(ROOT/'DepictQA'/cfg['model'][k])
    cfg['train']['max_tokens']=256
    original=LlamaForCausalLM.from_pretrained
    def half_loader(*args,**kwargs):
        kwargs.update(torch_dtype=torch.float16,low_cpu_mem_usage=True)
        return original(*args,**kwargs)
    torch.set_num_threads(4)
    with patch.object(LlamaForCausalLM,'from_pretrained',side_effect=half_loader):
        model=DepictQA(EasyDict(cfg),training=False)
    checkpoint=torch.load(cfg['model']['delta_path'],map_location='cpu')
    status=model.load_state_dict(checkpoint,strict=False)
    if status.unexpected_keys:raise ValueError('unexpected author delta keys')
    model=model.half().cuda()
    names=[]
    for name,param in model.named_parameters():
        enabled=('lora_A' in name or 'lora_B' in name or (train_projector and name.startswith('vision_proj.')))
        param.requires_grad=enabled
        if enabled:param.data=param.data.float();names.append(name)
    if not names:raise ValueError('no trainable parameters')
    model.llm.config.use_cache=False
    # Pinned author Llama uses the pre-4.35 boolean hook, not the new HF signature.
    model.llm.model.model.gradient_checkpointing=True
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    # Cache only frozen, deterministic CLIP+abstractor features, never answers.
    signature=hashlib.sha256(json.dumps({'vision':digest(cfg['model']['vision_encoder_path']),'delta':digest(cfg['model']['delta_path']),'preprocess':cfg['vision_preprocess'],'source':digest(ROOT/'DepictQA/src/model/clip/clip.py')},sort_keys=True).encode()).hexdigest()
    memory={}
    def embedding(paths):
        if not paths[0]:return None
        if len(paths)!=1:raise ValueError('only one item per microbatch')
        path=str(paths[0]);key=hashlib.sha256((signature+digest(path)).encode()).hexdigest()
        if key not in memory:
            file=cache_dir/(key+'.pt')
            if file.exists():raw=torch.load(file,map_location='cpu')
            else:
                model.vision_encoder.eval();model.abstractor.eval()
                with torch.no_grad():
                    image=model.load_img([path],model.device).half()
                    raw=model.abstractor(model.vision_encoder.forward_patch_features(image)).detach().cpu()
                torch.save(raw,file)
            memory[key]=raw
        return model.vision_proj(memory[key].cuda())
    model.emb_img=embedding
    model.tokenize_conv=align_single_conversation(model.tokenize_conv)
    model.eval()
    identity={'config':cfg,'feature_cache_signature':signature,'trainable_names':names,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),'attention_protocol':'single_unpadded_conversation_all_visible_matching_native_generation_v2','frozen_vision_abstractor':True,'trainable_dtype':'float32','backbone_dtype':'float16','gradient_checkpointing':True,'source_commit':__import__('subprocess').check_output(['git','-C',str(ROOT/'DepictQA'),'rev-parse','HEAD'],text=True).strip()}
    return model,identity


def batch(row):
    return {'task_type':['quality_compare_noref'],'img_path':[None],'img_A_path':[row['image_A']],'img_B_path':[row['image_B']],'conversation':[[{'from':'human','value':row['prompt']},{'from':'gpt','value':row['answer']}]]}


def adapters(model):
    return {k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}


def set_adapters(model,state):
    import torch
    params=dict(model.named_parameters())
    expected={k for k,p in params.items() if p.requires_grad}
    if set(state)!=expected:raise ValueError('adapter key mismatch')
    with torch.no_grad():
        for k,value in state.items():params[k].copy_(value.to(params[k].device))


def smoke(args):
    import torch
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise ValueError('smoke output already completed')
    row=json.loads(Path(args.data).read_text())[0]
    start=time.monotonic();torch.manual_seed(929);torch.cuda.manual_seed_all(929)
    model,identity=load_model(args.cache);save_json(out/'model_identity.json',identity)
    params=[p for p in model.parameters() if p.requires_grad];initial=adapters(model)
    opt=torch.optim.AdamW(params,lr=1e-5);scaler=torch.cuda.amp.GradScaler(init_scale=256)
    model.train();model.vision_encoder.eval();model.abstractor.eval();torch.cuda.reset_peak_memory_stats()
    t=time.monotonic()
    with torch.cuda.amp.autocast():loss,_=model(batch(row))
    assert torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.unscale_(opt)
    grad=torch.nn.utils.clip_grad_norm_(params,1.0);assert torch.isfinite(grad)
    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
    updated=adapters(model);changed=[k for k in updated if not torch.equal(updated[k],initial[k])]
    assert changed,'optimizer did not update any parameters'
    ckpt=out/'smoke_adapter.pt';torch.save(updated,ckpt)
    model.eval()
    with torch.no_grad(),torch.cuda.amp.autocast():before=float(model(batch(row))[0])
    set_adapters(model,initial);set_adapters(model,torch.load(ckpt,map_location='cpu'))
    with torch.no_grad(),torch.cuda.amp.autocast():after=float(model(batch(row))[0])
    assert before==after,'reload changed deterministic loss'
    result={'status':'ok','initial_train_loss':float(loss),'reload_eval_loss':after,'reload_exact':True,'changed_tensor_count':len(changed),'trainable_parameters':identity['trainable_parameters'],'gradient_norm_before_clip':float(grad),'update_and_reload_seconds':time.monotonic()-t,'total_seconds':time.monotonic()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'checkpoint':str(ckpt.resolve()),'checkpoint_sha256':digest(ckpt),'proof_scope':'single update/load connectivity only; not effective training'}
    save_json(out/'result.json',result);print(json.dumps(result),flush=True)


def canonical_prediction(text):
    import re
    text=text.strip().replace('</s>','').strip()
    if re.match(r'^(?:Image\s+)?A(?:\b|$)',text,re.I):return 'A'
    if re.match(r'^(?:Image\s+)?B(?:\b|$)',text,re.I):return 'B'
    if re.match(r'^(?:Tie|Equal|Same)(?:\b|$)',text,re.I):return 'Tie'
    if re.match(r'^(?:Uncertain|Cannot|Unable)(?:\b|$)',text,re.I):return 'Uncertain'
    return 'invalid'


def evaluate_rows(model,rows,out,tag):
    import torch
    from collections import defaultdict,Counter
    model.eval();records=[];start=time.monotonic()
    for i,row in enumerate(rows):
        with torch.no_grad(),torch.cuda.amp.autocast():
            text=model.generate({'query':[row['prompt']],'img_path':[None],'img_A_path':[row['image_A']],'img_B_path':[row['image_B']],'temperature':0.0,'top_p':.9,'max_new_tokens':12,'task_type':'quality_compare_noref','output_prob_id':False,'output_confidence':False,'sentence_model':None})[0][0]
        prediction=canonical_prediction(text)
        records.append({k:row.get(k,'') for k in ['id','pair_id','group_id','sample_id','kind','split','answer','order','label_origin']}|{'prediction':prediction,'raw_answer':text,'correct':prediction==row['answer']})
        if (i+1)%32==0:print('eval',tag,i+1,'/',len(rows),flush=True)
    classes={c:[r for r in records if r['answer']==c] for c in ['A','B','Tie','Uncertain']}
    class_acc={c:sum(r['correct'] for r in rs)/len(rs) for c,rs in classes.items() if rs}
    pairs=defaultdict(list)
    for r in records:pairs[r['pair_id']].append(r)
    def normalize(r):
        q=r['prediction']
        return ('left' if (q=='A')==(r['order']=='forward') else 'right') if q in ['A','B'] else q
    valid_pairs=[rs for rs in pairs.values() if len(rs)==2]
    consistent=sum(len({normalize(r) for r in rs})==1 and all(r['prediction']!='invalid' for r in rs) for rs in valid_pairs)
    group_acc={g:sum(r['correct'] for r in records if r['group_id']==g)/sum(r['group_id']==g for r in records) for g in {r['group_id'] for r in records}}
    summary={'tag':tag,'n_qa':len(records),'n_pairs':len(pairs),'n_groups':len(group_acc),'accuracy':sum(r['correct'] for r in records)/len(records),'macro_class_accuracy':sum(class_acc.values())/len(class_acc),'class_accuracy':class_acc,'group_equal_accuracy':sum(group_acc.values())/len(group_acc),'group_accuracy':group_acc,'swap_consistent_pairs':consistent,'swap_pair_count':len(valid_pairs),'swap_consistency':consistent/len(valid_pairs) if valid_pairs else None,'invalid':sum(r['prediction']=='invalid' for r in records),'predicted_counts':dict(Counter(r['prediction'] for r in records)),'seconds':time.monotonic()-start}
    save_json(Path(out)/(tag+'_predictions.json'),records);save_json(Path(out)/(tag+'_summary.json'),summary);print('EVAL_SUMMARY',json.dumps(summary),flush=True)
    return summary


def verify_rows(train,val):
    train_groups=set();val_groups=set()
    for rows,groups in [(train,train_groups),(val,val_groups)]:
        for r in rows:
            groups.add(r['group_id'])
            if r.get('other_group_id'):groups.add(r['other_group_id'])
            if r.get('domain','street')!='street' or r.get('split') not in ['train','val']:raise ValueError('invalid training/validation scope')
            for name in ['image_A','image_B']:
                if digest(r[name])!=r[name+'_sha256']:raise ValueError('input changed')
    if train_groups&val_groups:raise ValueError('source groups leak between train and val')
    if any(r['split']!='train' for r in train) or any(r['split']!='val' for r in val):raise ValueError('split mismatch')
    train_images={r[k+'_sha256'] for r in train for k in ['image_A','image_B']};val_images={r[k+'_sha256'] for r in val for k in ['image_A','image_B']}
    if train_images&val_images:raise ValueError('exact input duplicate across splits')


def train_run(args):
    import torch,math
    out=Path(args.out)
    if out.exists():raise ValueError('training run directory must be new')
    train=json.loads(Path(args.data).read_text());val=json.loads(Path(args.val).read_text());verify_rows(train,val)
    out.mkdir(parents=True);random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    cfg=vars(args).copy();cfg.update(train_sha256=digest(args.data),val_sha256=digest(args.val),source_sha256=digest(__file__),selection='highest validation macro class accuracy; ties retain earlier epoch; baseline never updated',effective_batch=args.accumulation)
    save_json(out/'training_config.json',cfg)
    model,identity=load_model(args.cache);save_json(out/'model_identity.json',identity)
    baseline=evaluate_rows(model,val,out,'baseline')
    baseline_weights=adapters(model);torch.save(baseline_weights,out/'initial_adapter.pt')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=256);steps=0;history=[];best=-1.;best_epoch=None;total_start=time.monotonic()
    total_updates=args.epochs*math.ceil(len(train)/args.accumulation)
    for epoch in range(1,args.epochs+1):
        order=list(range(len(train)));random.Random(args.seed+epoch).shuffle(order)
        model.train();model.vision_encoder.eval();model.abstractor.eval();optimizer.zero_grad(set_to_none=True);losses=[];epoch_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for pos,index in enumerate(order):
            row=train[index];group_start=(pos//args.accumulation)*args.accumulation;denom=min(args.accumulation,len(order)-group_start)
            with torch.cuda.amp.autocast():loss,_=model(batch(row))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            losses.append(float(loss.detach()));scaler.scale(loss/denom).backward()
            if (pos+1)%args.accumulation==0 or pos+1==len(order):
                scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()<old_scale:raise FloatingPointError('optimizer update skipped')
                optimizer.zero_grad(set_to_none=True);steps+=1
                progress=steps/total_updates;lr=args.lr*max(.1,.5*(1+math.cos(math.pi*progress)))
                for group in optimizer.param_groups:group['lr']=lr
                rec={'epoch':epoch,'microbatches':pos+1,'optimizer_step':steps,'loss':sum(losses[-denom:])/denom,'gradient_norm':float(norm),'lr':lr,'seconds':time.monotonic()-total_start}
                with (out/'training_steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                if steps%10==0:print('TRAIN',json.dumps(rec),flush=True)
        ckpt=out/('epoch_'+str(epoch)+'.pt');state=adapters(model);torch.save(state,ckpt)
        changed=sum(not torch.equal(state[k],baseline_weights[k]) for k in state);assert changed
        allocated=torch.cuda.max_memory_allocated()/1024**2;reserved=torch.cuda.max_memory_reserved()/1024**2
        summary=evaluate_rows(model,val,out,'epoch_'+str(epoch));summary.update(epoch=epoch,optimizer_steps=steps,train_loss=sum(losses)/len(losses),train_seconds=time.monotonic()-epoch_start-summary['seconds'],peak_allocated_mib=allocated,peak_reserved_mib=reserved,changed_tensors=changed,checkpoint_sha256=digest(ckpt))
        history.append(summary);save_json(out/'history.json',history)
        if summary['macro_class_accuracy']>best:
            best=summary['macro_class_accuracy'];best_epoch=epoch;torch.save(state,out/'best_adapter.pt')
    result={'status':'completed','epochs':args.epochs,'optimizer_updates':steps,'train_qa':len(train),'val_qa':len(val),'best_epoch':best_epoch,'best_val_macro_accuracy':best,'baseline_val_macro_accuracy':baseline['macro_class_accuracy'],'best_adapter_sha256':digest(out/'best_adapter.pt'),'training_validation_seconds':time.monotonic()-total_start,'human_preference_validation':False,'promotion':'not deployed; auditable constructed-label study only','study_scope':args.study_scope}
    save_json(out/'result.json',result);print('TRAIN_COMPLETE',json.dumps(result),flush=True)


def evaluation_run(args):
    import torch
    rows=json.loads(Path(args.data).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    model,identity=load_model(args.cache);save_json(out/'model_identity.json',identity)
    evaluate_rows(model,rows,out,'baseline')
    state=torch.load(args.adapter,map_location='cpu');set_adapters(model,state)
    evaluate_rows(model,rows,out,'trained')
    save_json(out/'identity.json',{'adapter':str(Path(args.adapter).resolve()),'adapter_sha256':digest(args.adapter),'data_sha256':digest(args.data),'source_sha256':digest(__file__),'new_model_selection_forbidden':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','train','evaluate']);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--cache',required=True);p.add_argument('--val');p.add_argument('--adapter');p.add_argument('--epochs',type=int,default=3);p.add_argument('--accumulation',type=int,default=4);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--seed',type=int,default=929);p.add_argument('--study-scope',default='old_development_sources')
    a=p.parse_args()
    if a.action=='smoke':smoke(a)
    elif a.action=='train':
        if not a.val or a.epochs<1 or a.accumulation<1:p.error('training requires validation and positive limits')
        train_run(a)
    else:
        if not a.adapter:p.error('evaluation requires adapter')
        evaluation_run(a)
if __name__=='__main__':main()
