"""Post-selection diagnostics with fixed inputs and unchanged per-arm rules."""
import argparse
import json
from pathlib import Path
import torch
from .vlm_training import (load_model, set_adapters, evaluate_rows, canonical_prediction,
                           save_json, digest, batch)
from .vlm_choice_eval import score_options
from pipeline.prompts import depictqa_compare_prompt


def unlabeled_generation(model, rows, out, tag):
    records = []
    model.eval()
    for row in rows:
        with torch.no_grad(), torch.cuda.amp.autocast():
            answer = model.generate(dict(query=[row['prompt']], img_path=[None],
                img_A_path=[row['image_A']], img_B_path=[row['image_B']],
                temperature=0., top_p=.9, max_new_tokens=12,
                task_type='quality_compare_noref', output_prob_id=False,
                output_confidence=False, sentence_model=None))[0][0]
        records.append({k: row.get(k) for k in ['id','pair_id','order','domain','sample_id']}
                       | dict(raw_answer=answer, prediction=canonical_prediction(answer),
                              correct=None, human_label=None))
    save_json(out / (tag + '_predictions.json'), records)
    return records


def frozen_gate_decisions(records):
    """Accept a fixed right-hand candidate only when BOTH orderings prefer it."""
    pair_ids = sorted({r['pair_id'] for r in records})
    return [dict(pair_id=pid, accept_fixed_right_candidate=
        {r['order']:r['prediction'] for r in records if r['pair_id']==pid}
        == {'forward':'B', 'reverse':'A'} and sum(r['pair_id']==pid for r in records)==2,
        limitation='offline frozen-candidate replay only; no replanning or tool rerun')
        for pid in pair_ids]


def prefix_parity(model, row):
    """Verify real fused visual/text prefixes, beyond a mock tokenizer check."""
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        ids, targets, mask=model.tokenize_conv(batch(row)['conversation'],'quality_compare_noref')
        embeds, full_targets, full_mask=model.fuse_vision_iqa(
            vision_embs=[None,model.emb_img([row['image_A']]),model.emb_img([row['image_B']])],
            input_ids=ids,tgt_ids=targets,attn_mask=mask,task_type='quality_compare_noref')
        boundary=int(torch.where(full_targets[0]!=-100)[0][0])
        gen_embeds, gen_mask=model.get_generate_embs(dict(query=[row['prompt']],
            img_path=[None],img_A_path=[row['image_A']],img_B_path=[row['image_B']],
            task_type='quality_compare_noref'))
        equal_embeds=torch.equal(embeds[:,:boundary],gen_embeds)
        equal_masks=torch.equal(full_mask[:,:boundary],gen_mask)
        if not equal_embeds or not equal_masks:
            raise ValueError('training/generation fused prefix mismatch')
        train_logits=model.llm(inputs_embeds=embeds,attention_mask=full_mask,
            return_dict=True).logits[0,boundary-1]
        gen_logits=model.llm(inputs_embeds=gen_embeds,attention_mask=gen_mask,
            return_dict=True).logits[0,-1]
    return dict(fused_prefix_equal=equal_embeds, attention_mask_equal=equal_masks,
        prefix_length=boundary, first_token_argmax_equal=bool(train_logits.argmax()==gen_logits.argmax()),
        first_token_train=int(train_logits.argmax()), first_token_generation=int(gen_logits.argmax()),
        max_logit_difference=float((train_logits-gen_logits).abs().max()),
        note='FP16 matrix shape may cause small rounding differences; no response-token attention leakage')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--mixed', required=True)
    p.add_argument('--controlled', required=True)
    p.add_argument('--fresh')
    p.add_argument('--fresh-test')
    a=p.parse_args(); out=Path(a.out)
    if out.exists(): raise ValueError('evaluation directory must be new')
    cfg=json.loads(Path(a.protocol).read_text()); datasets={}
    for name, spec in cfg['datasets'].items():
        if digest(spec['path'])!=spec['sha256']: raise ValueError('evaluation input list changed')
        datasets[name]=json.loads(Path(spec['path']).read_text())
        for row in datasets[name]:
            for key in ['image_A','image_B']:
                if digest(row[key])!=row[key+'_sha256']: raise ValueError('evaluation image changed')
    if bool(a.fresh)!=bool(a.fresh_test): raise ValueError('fresh adapter and test must be supplied together')
    if a.fresh_test:
        datasets['new_test']=json.loads(Path(a.fresh_test).read_text())
        for row in datasets['new_test']:
            if row['split']!='final_test': raise ValueError('wrong final test split')
            for key in ['image_A','image_B']:
                if digest(row[key])!=row[key+'_sha256']:raise ValueError('final test image changed')
    out.mkdir(parents=True)
    torch.manual_seed(929); torch.cuda.manual_seed_all(929)
    model, identity=load_model(a.cache)
    save_json(out/'model_identity.json',identity)
    adapters={'baseline':None, 'mixed':a.mixed, 'controlled':a.controlled}
    if a.fresh: adapters['fresh']=a.fresh
    for arm, path in adapters.items():
        folder=out/arm; folder.mkdir()
        if path: set_adapters(model,torch.load(path,map_location='cpu'))
        save_json(folder/'identity.json',dict(adapter=path,
            adapter_sha256=digest(path) if path else None,
            protocol_sha256=digest(a.protocol), source_sha256=digest(__file__),
            fresh_test_sha256=digest(a.fresh_test) if a.fresh_test else None,
            choice_source_sha256=digest(Path(__file__).with_name('vlm_choice_eval.py')),
            train_source_sha256=digest(Path(__file__).with_name('vlm_training.py')),
            loaded_in_fresh_process=True, model_selection_forbidden=True))
        audit_rows=datasets['new_test'] if a.fresh_test and arm in ['baseline','fresh'] else datasets['val']
        save_json(folder/'prefix_parity.json',prefix_parity(model,next(r for r in audit_rows if r['answer']=='Tie')))
        if arm!='fresh':
            score_options(model,datasets['val'],folder,'val')
            native=[dict(r,prompt=depictqa_compare_prompt) for r in datasets['val'] if r['answer'] in ['A','B']]
            evaluate_rows(model,native,folder,'val_native')
        if a.fresh and arm in ['mixed','controlled']:
            print('ARM_COMPLETE',arm,'old development comparison only',flush=True)
            continue
        for domain in (['new_test'] if a.fresh_test else [])+['generic','tianlian']:
            evaluate_rows(model,datasets[domain],folder,domain)
            score_options(model,datasets[domain],folder,domain)
            if domain=='new_test':
                native=[dict(r,prompt=depictqa_compare_prompt) for r in datasets[domain] if r['answer'] in ['A','B']]
                evaluate_rows(model,native,folder,'new_test_native')
        score_options(model,datasets['human_pending'],folder,'human_pending')
        unlabeled_generation(model,datasets['human_pending'],folder,'human_pending')
        rec=unlabeled_generation(model,datasets['frozen_trajectory'],folder,'frozen_trajectory')
        save_json(folder/'frozen_gate_decisions.json',frozen_gate_decisions(rec))
        print('ARM_COMPLETE',arm,flush=True)
    save_json(out/'COMPLETE.json',dict(status='completed',arms=list(adapters),
        protocol_sha256=digest(a.protocol),no_model_reselection=True))

if __name__=='__main__': main()
