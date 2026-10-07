"""Construct auditable reference-proxy preferences between actual tool outputs.

The reference is a pre-perturbation observation, potentially already degraded.
These are objective pseudo labels for a controlled incremental restoration task,
not human preferences or evidence of correct restoration of real rain.
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .manifest_io import read_csv, write_csv, sha256
from .experiment_support import atomic_json
from .prepare_vlm_data import PROMPT
from .score_manifest import Metrics, pixels


def consensus(left, right, margins):
    improvements = {'psnr':left['psnr']-right['psnr'],
                    'ssim':left['ssim']-right['ssim'],
                    'lpips':right['lpips']-left['lpips']}
    if all(improvements[k] >= margins[k] for k in margins):
        return 'left'
    if all(improvements[k] <= -margins[k] for k in margins):
        return 'right'
    return None


def qa_rows(pair):
    rows=[]
    for order in ['forward','reverse']:
        a,b=(pair['left'],pair['right']) if order=='forward' else (pair['right'],pair['left'])
        label=pair['canonical_label']
        if label in ['left','right']:
            label='A' if (label=='left')==(order=='forward') else 'B'
        rows.append(dict(pair,id=pair['pair_id']+'__'+order,order=order,
                         image_A=a,image_B=b,image_A_sha256=sha256(a),image_B_sha256=sha256(b),
                         prompt=PROMPT,answer=label,human_reviewed=False))
    return rows


def prepare(manifest, results, protocol, out, small_sources, weights):
    out=Path(out)
    if out.exists():
        raise ValueError('choose a fresh supervised-data output directory')
    out.mkdir(parents=True)
    cfg=json.loads(Path(protocol).read_text())
    m=cfg['admission_margin'];margins={'psnr':m['psnr_db'],'ssim':m['ssim'],'lpips':m['lpips']}
    sources=read_csv(manifest);restored=read_csv(results);engine=Metrics('cpu',weights)
    metric_rows=[];pairs=[];pending=[]
    def pair(r,left,right,kind,label,origin,**kw):
        return dict(pair_id=r['sample_id']+'__'+kind,sample_id=r['source_sample_id'],
                    group_id=r['group_id'],split=r['training_split'],domain=r['domain'],
                    left=left,right=right,kind=kind,canonical_label=label,label_origin=origin,**kw)
    for r in sources:
        candidates=[('input',r['input_path'])]+[(q['method'],q['output_path']) for q in restored
            if q['sample_id']==r['sample_id'] and q['status']=='ok']
        ref=pixels(r['reference_path']);scores={};arrays={}
        for name,path in candidates:
            x=pixels(path);arrays[name]=x
            scores[name]={metric:engine.score(metric,x,ref) for metric in margins}
            metric_rows.append(dict(sample_id=r['sample_id'],method=name,**scores[name]))
        for (a,ap),(b,bp) in itertools.combinations(candidates,2):
            label=consensus(scores[a],scores[b],margins)
            if np.array_equal(arrays[a],arrays[b]):
                label='Tie';origin='exact_pixel_identity'
            else:
                origin='controlled_reference_multimetric_pseudo_label'
            item=pair(r,ap,bp,'restoration_'+a+'_vs_'+b,label,origin,
                      basis='PSNR/SSIM/LPIPS consensus relative to pre-overlay observation; not human preference')
            if label is None:
                item.update(label_origin='pending_human_review',basis='near tie or metric disagreement; not supervised Uncertain')
                pending.append(item)
            else:
                pairs.append(item)
        write_csv(out/'candidate_metrics.csv',metric_rows)
        print('prepared preferences',r['sample_id'],flush=True)
    # Sparse controls: one source per group, rather than many easy cases per image.
    for split in ['train','val']:
        eligible=[r for r in sources if r['training_split']==split]
        for group in sorted({r['group_id'] for r in eligible}):
            r=next(x for x in eligible if x['group_id']==group)
            pairs.append(pair(r,r['reference_path'],r['reference_path'],'identity','Tie','exact_pixel_identity'))
            others=[q for q in eligible if q['group_id']!=group and q['scene_id']!=r['scene_id']]
            if others:
                other=others[0]
                pairs.append(pair(r,r['reference_path'],other['reference_path'],'noncorrespondence',
                                  'Uncertain','source_metadata',other_group_id=other['group_id'],
                                  other_sample_id=other['source_sample_id']))
    all_rows=[row for p in pairs for row in qa_rows(p)]
    small_ids={r['sample_id'] for r in read_csv(small_sources) if r['split']=='train'}
    sets={'train_large':[r for r in all_rows if r['split']=='train'],
          'val':[r for r in all_rows if r['split']=='val']}
    sets['train_small']=[r for r in sets['train_large'] if r['sample_id'] in small_ids
                         and (not r.get('other_sample_id') or r['other_sample_id'] in small_ids)]
    summary={}
    for name,rows in sets.items():
        atomic_json(out/(name+'.json'),rows)
        used={r['sample_id'] for r in rows}|{r['other_sample_id'] for r in rows if r.get('other_sample_id')}
        summary[name]={'qa':len(rows),'pairs':len(rows)//2,'actual_source_images':len(used),
                       'groups':len({r['group_id'] for r in rows}),
                       'labels':dict(Counter(r['answer'] for r in rows)),
                       'origins':dict(Counter(r['label_origin'] for r in rows))}
    atomic_json(out/'pending_review.json',pending);atomic_json(out/'pairs.json',pairs)
    atomic_json(out/'summary.json',dict(sets=summary,unlabeled_pairs=len(pending),
        independent_sources_available=len(sources),new_directory_sources_received=0,
        limitations=['Proxy-supervised pilot, not the planned new-source expansion',
                     'Underlying source image is not verified rain-free ground truth',
                     'No human judgments; disagreement withheld, not labeled Uncertain'],
        preparation_source_sha256=sha256(__file__),protocol_sha256=sha256(protocol),
        manifest_sha256=sha256(manifest),results_sha256=sha256(results)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['manifest','results','protocol','out','small-sources','weights']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.manifest,a.results,a.protocol,a.out,a.small_sources,a.weights)
