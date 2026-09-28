"""Bounded resumable manifest runner. B0 is an explicitly named approximation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import time
import traceback
from .experiment_support import code_hashes, failure_category
from .text_planner import ModelAccessError, MODEL
from .manifest_io import load_manifest, read_csv, write_csv, sha256
from .score_manifest import pixels


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['B0_registry_no_retrieval','B1_current','restormer_smoke'],default='B1_current')
    p.add_argument('--mode',choices=['auto','manual'],default='auto'); p.add_argument('--plan',default='')
    p.add_argument('--domain'); p.add_argument('--role'); p.add_argument('--split',choices=['dev','test'],default='dev'); p.add_argument('--limit',type=int,default=4)
    p.add_argument('--evaluator',choices=['depictqa'],default='depictqa'); p.add_argument('--llm-config',default='config.yml')
    p.add_argument('--schedule',choices=['on','off']); p.add_argument('--selector',choices=['current','registry'])
    p.add_argument('--memory-read',choices=['off','frozen'],default='frozen'); p.add_argument('--memory-snapshot'); p.add_argument('--memory-update',action='store_true'); p.add_argument('--no-event-log',action='store_true')
    p.add_argument('--max-tool-calls',type=int,default=8); p.add_argument('--max-llm-calls',type=int,default=20); p.add_argument('--seed',type=int,default=2700); p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--staged-gpu', help='allocated GPU UUID for sequential live local services')
    p.add_argument('--catalog', help='JSON mapping every task to available tools; staging only')
    a=p.parse_args()
    if a.catalog and not a.staged_gpu: p.error('--catalog requires --staged-gpu')
    if a.staged_gpu and a.method=='restormer_smoke': p.error('staging requires an agent method')
    if a.mode=='manual' and not a.plan: p.error('manual mode requires --plan')
    if a.mode=='auto' and a.plan: p.error('auto mode cannot receive a plan')
    if a.limit<1 or a.max_tool_calls<1 or a.max_llm_calls<1: p.error('limits must be positive')
    if a.memory_update and a.no_event_log: p.error('online memory needs event logging')
    if a.memory_update and a.memory_read=='off': p.error('online update requires reads')
    rows=[r for r in load_manifest(a.manifest) if r['role']!='base' and r['split']==a.split and
          (not a.domain or r['domain']==a.domain) and (not a.role or r['role']==a.role)]
    # Interleave domains and real/synthetic, giving a representative bounded smoke set.
    rows=sorted(rows,key=lambda r:(r['domain'],r['role'],r['sample_id']))
    buckets={}
    for r in rows: buckets.setdefault((r['domain'],r['role']),[]).append(r)
    selected=[]
    while len(selected)<a.limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected)<a.limit: selected.append(buckets[key].pop(0))
    if not selected: p.error('selection is empty')
    out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    config=vars(a).copy(); config['manifest_sha256']=sha256(a.manifest)
    import yaml
    llm_cfg=yaml.safe_load(Path(a.llm_config).read_text())
    config['planner']={k:v for k,v in llm_cfg['GPT'].items() if 'KEY' not in k and 'SECRET' not in k}
    if config['planner']['MODEL'] != MODEL: raise ValueError('wrong text planner model')
    config['code']=code_hashes()
    config['catalog_sha256']=sha256(a.catalog) if a.catalog else None
    config['memory_sha256']=sha256(a.memory_snapshot) if a.memory_snapshot else None
    config['input_hashes']={r['sample_id']:sha256(r['input_path']) for r in selected}
    config['policies']={n:sha256(Path('memory')/n) for n in ['schedule_rules.json','tool_profiles.json']}
    config['tool_inventory']={str(p):sha256(p) for p in Path('executor').glob('*.py')}
    config['weights_and_third_party_sources']={}
    for root in Path('executor').glob('*/tools'):
        for toolroot in root.iterdir():
            if not toolroot.is_dir(): continue
            for artifact in toolroot.rglob('*'):
                if artifact.is_file() and artifact.suffix in {'.py','.yml','.yaml','.pth','.pt','.ckpt','.h5','.npz'}:
                    stat=artifact.stat()
                    config['weights_and_third_party_sources'][str(artifact)]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                        'sha256':sha256(artifact) if artifact.suffix in {'.py','.yml','.yaml'} else None}
    config['model_identity_note']='LLM model/settings; tool weight path+size+mtime; third-party source hashes. Fixed-tool studies additionally hash tested weights.'

    config['selector']=a.selector or ('registry' if a.method.startswith('B0') else 'current')
    config['schedule']=a.schedule or ('off' if a.method.startswith('B0') else 'on')
    config['memory_state']='snapshot' if a.memory_snapshot else 'cold_start'
    cfg=out/'run_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config: raise ValueError('resume config mismatch; choose new directory')
    cfg.write_text(json.dumps(config,indent=2))
    # Copy common policies once. Never read or update production runtime memory.
    for name in ['schedule_rules.json','tool_profiles.json']:
        dest=out/name
        if not dest.exists(): shutil.copy2(Path('memory')/name,dest)
    snapshot=out/'statistics_snapshot.sqlite3'
    if a.memory_snapshot and not snapshot.exists():
        with sqlite3.connect('file:'+str(Path(a.memory_snapshot).resolve())+'?mode=ro',uri=True) as src, sqlite3.connect(snapshot) as dst: src.backup(dst)
    if a.memory_update and snapshot.exists() and not (out/'events.sqlite3').exists():
        with sqlite3.connect('file:'+str(snapshot)+'?mode=ro',uri=True) as src, sqlite3.connect(out/'events.sqlite3') as dst: src.backup(dst)
    results_path=out/'results.csv'; results=read_csv(results_path) if results_path.exists() else []
    write_csv(out/'selected_manifest.csv',selected)
    def alarm(*unused): raise TimeoutError('per-sample time budget exceeded')
    signal.signal(signal.SIGALRM,alarm)
    for r in selected:
        old=next((q for q in results if q['sample_id']==r['sample_id']),None)
        if old and old.get('input_sha256')==sha256(r['input_path']) and old['status']=='ok' and Path(old['output_path']).exists() and sha256(old['output_path'])==old.get('output_sha256'):
            pixels(old['output_path']); continue
        results=[q for q in results if q['sample_id']!=r['sample_id']]
        row={k:r.get(k,'') for k in ['sample_id','domain','role','split','group_id','condition','input_path','reference_path']}
        row.update(input_sha256=sha256(r['input_path']),method=a.method,mode=a.mode,status='failed',output_path='',error='',duration_seconds='',n_invocations='',topk_decisions='',ranking_decisions='')
        started=time.monotonic(); work=out/'runs'/r['sample_id']; work.mkdir(parents=True,exist_ok=True)
        agent=None; controller=None
        try:
            import random,numpy as np,torch
            random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
            torch.backends.cudnn.benchmark=False
            pixels(r['input_path']); signal.alarm(a.timeout)
            if a.method=='restormer_smoke':
                from executor import executor
                inp=work/'input'; inp.mkdir(exist_ok=True); shutil.copy2(r['input_path'],inp/'input.png')
                dest=work/('output_'+str(time.time_ns())); dest.mkdir()
                tool=next(t for t in executor.toolbox_router['deraining'] if t.tool_name=='restormer')
                res=tool(inp,dest,silent=True); output=res.output_path; row['n_invocations']=1
                row['mode']='manual_single_tool'
            else:
                from .experiment_agent import ExperimentAgent
                agent=ExperimentAgent(input_path=Path(r['input_path']),output_dir=work,llm_config_path=Path(a.llm_config),
                    evaluate_degradation_by=a.evaluator,reflect_by=a.evaluator,
                    with_retrieval=config['schedule']=='on',schedule_experience_path=out/'schedule_rules.json',
                    episode_store_path=out/'events.sqlite3',tool_profile_path=out/'tool_profiles.json',
                    selector_policy=config['selector'],memory_read=a.memory_read,memory_snapshot=snapshot if snapshot.exists() else None,
                    memory_update=a.memory_update,log_events=not a.no_event_log,max_tool_calls=a.max_tool_calls,max_llm_calls=a.max_llm_calls,random_seed=a.seed,silent=True)
                if a.staged_gpu:
                    from .staged_service import StagedServices, attach_staging
                    controller=StagedServices(work/'local_services',a.staged_gpu)
                    attach_staging(agent,controller,json.loads(Path(a.catalog).read_text()) if a.catalog else None)
                agent.run(plan=a.plan.split(',') if a.mode=='manual' else None)
                output=agent.work_dir/'result.png'
            if pixels(output).shape!=pixels(r['input_path']).shape: raise ValueError('output size differs; no scoring resize permitted')
            row.update(status='ok',output_path=str(output),output_sha256=sha256(output))
        except Exception as exc:
            row['error']=type(exc).__name__+': '+str(exc); row['failure_category']=failure_category(exc)
            (work/'error.txt').write_text(traceback.format_exc())
        finally:
            signal.alarm(0)
            if controller:
                try: controller.release()
                except Exception as cleanup_error:
                    row.update(status='failed',error='service cleanup failed: '+str(cleanup_error))
                row['local_model_starts']=controller.starts
                row['deployment']='single_gpu_live_staged'
            if agent:
                (work/'agent_summary.json').write_text(json.dumps(agent.work_mem,default=str,indent=2))
                ranks=agent.work_mem['tool_rankings']; row['n_invocations']=agent.work_mem['n_invocations']
                row['llm_calls']=agent.llm_calls; row['local_evaluation_calls']=agent.local_evaluation_calls; row['actual_tool_calls']=agent.tool_calls; row['ranking_decisions']=len(ranks); row['topk_decisions']=sum(len(q['selected_tools'])<len(q['ranked_tools']) for q in ranks)
                row['execution_path']=json.dumps(agent.work_mem['execution_path']); row['plan']=json.dumps(agent.work_mem['plan'])
            row['duration_seconds']=round(time.monotonic()-started,3); results.append(row); write_csv(results_path,results)
            print(row['sample_id'],row['status'],row['duration_seconds'],flush=True)
            if row.get('failure_category')=='api_access': break

if __name__=='__main__': main()
