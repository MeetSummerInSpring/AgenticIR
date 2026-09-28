"""Small study tables and contact sheets from real outputs; no expert scores."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .manifest_io import read_csv,write_csv
from .experiment_support import atomic_json


def main():
    root=Path('experiments/midterm_v2');metrics=read_csv(root/'scores/metrics_per_sample.csv');results=read_csv(root/'results.csv');manifest=read_csv(root/'manifest.csv');index={r['sample_id']:r for r in manifest}
    groups=defaultdict(list)
    for r in metrics:groups[(r['domain'],r['role'],r['condition'].split('/')[0],r['method'],r['metric'])].append(r)
    table=[];harms=[]
    for key,rows in groups.items():
        valid=[r for r in rows if r['status']=='ok'];source=defaultdict(list);bases=defaultdict(list)
        for r in valid:
            source[r['group_id']].append(float(r['delta']));bases[r['base_id'] or r['sample_id']].append(float(r['delta']))
            is_harm=float(r['improvement'])<0
            if is_harm:harms.append({**{k:r[k] for k in ['sample_id','domain','role','condition','method','metric','input_score','output_score','delta','improvement']},'interpretation':'metric worsening; not a human verdict'})
        mean=lambda v:float(np.mean(v)) if v else ''
        table.append(dict(zip(['domain','role','generator','method','metric'],key),n_total=len(rows),n_valid=len(valid),n_groups=len(source),n_bases=len(bases),
            input_mean=mean([float(r['input_score']) for r in valid]),output_mean=mean([float(r['output_score']) for r in valid]),delta_mean=mean([float(r['delta']) for r in valid]),
            group_equal_delta=mean([mean(v) for v in source.values()]),base_equal_delta=mean([mean(v) for v in bases.values()]),n_metric_worsened=sum(float(r['improvement'])<0 for r in valid)))
    write_csv(root/'study_summary.csv',table);write_csv(root/'metric_worsened.csv',harms)
    runtime=[]
    for method in sorted({r['method'] for r in results}):
        a=[r for r in results if r['method']==method]
        runtime.append({'method':method,'n':len(a),'n_success':sum(r['status']=='ok' for r in a),'calls':sum(int(r['n_tool_calls']) for r in a),
            'mean_wall_seconds':float(np.mean([float(r['duration_seconds']) for r in a])), 'mean_tool_seconds':float(np.mean([float(r['duration_tool_seconds']) for r in a])),
            'peak_device_memory_mib':max(float(r['peak_device_memory_mib']) for r in a)})
    write_csv(root/'runtime_summary.csv',runtime)
    # Primary fine synthetic + real, separated from coarse regression.
    primary=[r for r in manifest if r['role']=='real' or 'texture_screen_fine' in r['condition']]
    for domain in ['street','tianlian']:
        selected=[r for r in primary if r['domain']==domain];canvas=Image.new('RGB',(4*340,len(selected)*235),'#222222');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(selected):
            outputs={q['method']:q['output_path'] for q in results if q['sample_id']==r['sample_id'] and q['status']=='ok'}
            for i,(label,path) in enumerate([('input '+r['role'],r['input_path']),('Restormer',outputs.get('restormer','')),('MPRNet',outputs.get('mprnet','')),('reference' if r['reference_path'] else 'real: no reference',r['reference_path'])]):
                draw.text((i*340+3,j*235+4),label,fill='white')
                if path:
                    im=Image.open(path);im.thumbnail((336,206));canvas.paste(im,(i*340,j*235+26))
        canvas.save(root/(domain+'_restoration.jpg'),quality=93)
    atomic_json(root/'result_counts.json',{'inputs':len(manifest),'tool_records':len(results),'tool_success':sum(r['status']=='ok' for r in results),
        'metric_status_counts':dict(Counter(r['status'] for r in metrics)),'full_reference_missing_on_real':'intentional, no ground truth','primary_inputs':len(primary)})
if __name__=='__main__':main()
