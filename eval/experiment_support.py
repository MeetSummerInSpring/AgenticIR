"""Small shared helpers for resource evidence, resume identity, and failure labels."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import datetime
from .manifest_io import sha256


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str));tmp.replace(path)


def resources():
    def command(args):
        r=subprocess.run(args,capture_output=True,text=True);return {'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
    return {'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpus':command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits']),
        'processes':command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory,process_name','--format=csv,noheader,nounits'])}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def code_hashes():
    roots=['eval/manifest_io.py','eval/experiment_support.py','eval/tool_study.py','eval/run_manifest.py','eval/experiment_agent.py','eval/text_planner.py','eval/frozen_statistics.py','eval/prepare_holdout.py','eval/staged_service.py','eval/perception_probe.py','pipeline/iragent.py']
    files=[Path(p) for p in roots if Path(p).exists()]
    files += [Path(p) for p in ['eval/candidate_acceptance.py', 'eval/weather_context.py'] if Path(p).exists()]
    for root in ['utils','llm','executor']:
        files += [p for p in (p for p in Path(root).rglob('*.py') if 'tools' not in p.parts)]
    return {str(p):sha256(p) for p in sorted(set(files))}


def failure_category(error):
    text=str(error).lower()
    for names,category in [(['out of memory','cuda error: memory'],'gpu_oom'),(['no cuda','no gpu','resource busy'],'resource_unavailable'),
       (['filenotfound','no such file','does not exist','missing weight'],'missing_file'),(['modulenotfound','no module named','importerror'],'dependency'),
       (['allocationquota','403','401','modelaccesserror','insufficient_quota'],'api_access'),(['429'],'api_rate_limit'),
       (['connection refused','service unavailable'],'service_unavailable'),(['timeout','timed out'],'timeout'),
       (['jsondecode','invalid response','format','shape mismatch','expected rgb'],'format_error')]:
        if any(n in text for n in names):return category
    return 'execution_error'
