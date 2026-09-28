"""Build dev-only, scale/state-scoped tool evidence from local VLM feedback.
No reference path, synthetic label, or full-reference score enters SQLite.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from .manifest_io import read_csv,sha256,write_csv
from .experiment_support import atomic_json
from utils.episode_store import EpisodeStore
from utils.tool_selector import ToolSelector


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'provenance.json').exists():raise FileExistsError('calibration snapshot is immutable; select new directory')
    results=read_csv(a.results);labels=read_csv(a.labels)
    lookup={(r['sample_id'],r['method']):r for r in labels if r['status']=='ok'}
    groups={};exports=[]
    for r in results:
        if r['split']!='dev':raise ValueError('calibration rejects non-dev rows')
        inp=lookup.get((r['sample_id'],'input'));feedback=lookup.get((r['sample_id'],r['method']))
        if not inp:continue
        state=inp['answer'];key=(r.get('scale','unknown'),state,r['batch_fingerprint'])
        slug='s'+key[0]+'_'+state.replace(' ','_')+'_'+key[2][:10]
        if key not in groups:groups[key]=(slug,EpisodeStore(out/(slug+'.sqlite3')))
        store=groups[key][1]
        if r['status']=='ok' and feedback:
            quality_success=feedback['answer'] in ('very low','low');status='ok'
        elif r['status']!='ok':quality_success=None;status='execution_error'
        else:continue
        store.append_event(run_id='dev-calibration-'+__import__('hashlib').sha256((r['sample_id']+r['method']).encode()).hexdigest()[:20],event_type='tool_attempt',
            state={'scale_long_edge':int(r['scale']),'observed_rain_severity':state},action={'subtask':'deraining','tool':r['method']},
            transition={'duration_seconds':float(r.get('duration_tool_seconds') or r['duration_seconds'])},
            outcome={'status':status,'quality_success':quality_success,'severity_after':feedback['answer'] if feedback else None},
            provenance={'split':'dev','tool_batch_fingerprint':r['batch_fingerprint'],'feedback':'local_DepictQA rain model label','timing':'tool invocation incl conda startup; same input scale/state only'})
    # This is an offline selector diagnostic, not executed auto Top-K frequency.
    profile=json.loads(Path('memory/tool_profiles.json').read_text());profile['defaults']['exploration_rate']=0;profile['subtasks']['deraining']['max_tools']=1
    atomic_json(out/'diagnostic_profile.json',profile)
    for key,(slug,store) in groups.items():
        stats=store.get_tool_statistics('deraining');tools=[SimpleNamespace(tool_name=n) for n in sorted(stats)]
        selector=ToolSelector(profile_path=out/'diagnostic_profile.json',episode_store=store,seed=928)
        selected,diagnostics=selector.select('deraining',tools)
        exports.append({'scope':slug,'scale':key[0],'input_rain_state':key[1],'available_tools':json.dumps([t.tool_name for t in tools]),'selected':json.dumps([t.tool_name for t in selected]),
            'evidence_gated_pruning':len(selected)<len(tools),'statistics':json.dumps({k:asdict(v) for k,v in stats.items()}),'diagnostic_only':True})
        atomic_json(out/(slug+'_ranking.json'),diagnostics)
    write_csv(out/'selector_diagnostics.csv',exports)
    atomic_json(out/'provenance.json',{'results_sha256':sha256(a.results),'labels_sha256':sha256(a.labels),'dev_only':True,'full_reference_metrics_used':False,
        'observed_states_separated':True,'input_scales_separated':True,'tool_versions_separated':True,'automatic_pipeline_executed':False,
        'warning':'Diagnostics do not estimate deployed Top-K rate; use only a state/scale/tool-compatible snapshot in a future comparison.'})
if __name__=='__main__':main()
