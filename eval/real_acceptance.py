"""Offline acceptance diagnostic with an explicit original-image candidate.

These conservative input-fidelity checks are risk flags, not correctness labels.
No full-reference score, synthesis label, NIQE or updated memory drives selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .manifest_io import read_csv, write_csv
from .score_manifest import pixels

THRESHOLDS = {'roi_ssim_min': .95, 'roi_gradient_cosine_min': .9,
              'new_clipped_fraction_max': .005}
ROIS = {'tianlian': [.45, .20, .56, .97], 'street': [.05, .35, .95, .95]}


def preservation(x, y, roi):
    from skimage.metrics import structural_similarity
    if x.shape != y.shape:
        raise ValueError('shape mismatch')
    height, width = x.shape[:2]
    x0,y0,x1,y1 = [int(v*n) for v,n in zip(roi,[width,height,width,height])]
    a,b = x[y0:y1,x0:x1], y[y0:y1,x0:x1]
    if min(a.shape[:2]) < 11:
        raise ValueError('protected ROI too small')
    ga = np.concatenate([g.ravel() for g in np.gradient(a.mean(2))])
    gb = np.concatenate([g.ravel() for g in np.gradient(b.mean(2))])
    norm = float(np.linalg.norm(ga)*np.linalg.norm(gb))
    cosine = float(np.dot(ga,gb)/norm) if norm > 1e-10 else float(np.allclose(ga,gb))
    return {'roi_ssim':float(structural_similarity(a,b,data_range=1,channel_axis=2,win_size=11,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),
            'roi_gradient_cosine':cosine,
            'new_clipped_fraction':float(np.mean(np.any((y >= 1) & (x < 1),axis=2))),
            'mean_absolute_rgb_change':float(np.mean(abs(y-x)))}


def decide(labels, metrics):
    if len(labels) != 2 or not all(labels):
        return False, 'comparison_not_unanimous'
    if not all(np.isfinite(v) for v in metrics.values()):
        return False, 'invalid_content_metric'
    if metrics['roi_ssim'] < THRESHOLDS['roi_ssim_min'] or metrics['roi_gradient_cosine'] < THRESHOLDS['roi_gradient_cosine_min'] or metrics['new_clipped_fraction'] > THRESHOLDS['new_clipped_fraction_max']:
        return False, 'content_change_flag'
    return True, 'unanimous_and_preserved'


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--comparisons',required=True);p.add_argument('--severity',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    rows=read_csv(a.results); comp=read_csv(a.comparisons); severity=read_csv(a.severity)
    output=[]
    for r in rows:
        if r['status']!='ok':continue
        comparisons=[c for c in comp if c['sample_id']==r['sample_id'] and c['method']==r['method'] and c['status']=='ok']
        flags=[c['prefers_output']=='True' for c in comparisons]
        values=preservation(pixels(r['input_path']),pixels(r['output_path']),ROIS[r['domain']])
        accept,reason=decide(flags,values)
        levels=[s['answer'] for s in severity if s['sample_id']==r['sample_id'] and s['method']==r['method'] and s['status']=='ok']
        record={k:r.get(k,'') for k in ['sample_id','domain','role','split','group_id','base_id','condition','method']}
        record.update(values)
        record.update({
            'roi_normalized':json.dumps(ROIS[r['domain']]),'severity_only_accept':bool(levels and levels[0] in ['low','very low']),
            'single_order_accept':bool(comparisons and comparisons[0]['prefers_output']=='True'),
            'both_orders_accept':len(flags)==2 and all(flags),'guarded_accept':accept,'decision_reason':reason,
            'selected_path':r['output_path'] if accept else r['input_path'],
            'execution_mode':'offline cached single-tool accept/reject; all candidate costs already incurred'})
        output.append(record)
    write_csv(a.out,output)


if __name__=='__main__':main()
