"""Start/stop only owned local DepictQA processes; bind a selected allocated GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from .experiment_support import atomic_json,resources
from .manifest_io import sha256


def _port_open(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(('127.0.0.1',port))==0


def start_time(pid):
    return Path('/proc/'+str(pid)+'/stat').read_text().split()[21]


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop']);p.add_argument('--mode',choices=['severity','compare','dual'],required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',required=True);p.add_argument('--adapter',help='isolated comparison-only frozen adapter experiment');a=p.parse_args()
    if a.adapter and a.mode!='compare':p.error('--adapter is comparison-only')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);pidfile=out/(a.mode+'_owned_process.json')
    if a.action=='stop':
        if not pidfile.exists():raise RuntimeError('no owned process record')
        record=json.loads(pidfile.read_text());pid=record['pid']
        if not Path('/proc/'+str(pid)).exists():return
        if start_time(pid)!=record['start_time']:raise RuntimeError('PID reused; refusing to stop')
        cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes()
        if record['script'].encode() not in cmd:raise RuntimeError('process command changed; refusing to stop')
        os.killpg(pid,signal.SIGTERM)
        for _ in range(40):
            if not Path('/proc/'+str(pid)).exists():break
            time.sleep(.25)
        atomic_json(out/(a.mode+'_stopped.json'),{'pid':pid,'stopped_at':time.time()});return
    resource=resources();atomic_json(out/(a.mode+'_resource_before.json'),resource)
    gpu_rows=[line.split(',') for line in resource['gpus']['stdout'].splitlines()]
    allocated=os.environ.get('CUDA_VISIBLE_DEVICES')
    if allocated:
        allowed={token.strip() for token in allocated.split(',')}
        gpu_rows=[row for row in gpu_rows if row[0].strip() in allowed or row[1].strip() in allowed]
    target=next((row for row in gpu_rows if row[1].strip()==a.gpu),None)
    if target is None:raise RuntimeError('requested GPU UUID not in process-visible allocation')
    if float(target[-1])<17000:raise RuntimeError('insufficient free memory for this local VLM')
    if any(a.gpu in line for line in resource['processes']['stdout'].splitlines()):
        raise RuntimeError('selected GPU has existing processes; do not stop unknown processes')
    # Three-card deployments may keep the other route alive on a different UUID.
    ports=[5001,5002] if a.mode=='dual' else [5001 if a.mode=='severity' else 5002]
    for port in ports:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('port already in use; do not manage unknown service')
    script='src/app_eval.py' if a.mode=='severity' else 'src/app_comp.py';env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpu;env['OMP_NUM_THREADS']='4'
    root=Path('DepictQA').resolve()
    if 'serve(app, host="127.0.0.1"' not in (root/script).read_text():
        raise RuntimeError('apply installation/depictqa_localhost.patch before starting local evaluation services')
    adapter_sha256=sha256(a.adapter) if a.adapter else None
    log=(out/(a.mode+'_startup.log')).open('w')
    command=['/root/autodl-tmp/conda/envs/depictqa/bin/python','-u',script]
    cwd=root
    if a.adapter:
        script='eval.comparison_service';cwd=root.parent
        command=['/root/autodl-tmp/conda/envs/depictqa/bin/python','-u','-m',script,'--adapter',str(Path(a.adapter).resolve()),'--identity',str(out/'adapter_identity.json')]
    if a.mode=='dual':
        script='eval.dual_vision_service';cwd=root.parent
        command=['/root/autodl-tmp/conda/envs/depictqa/bin/python','-u','-m',script,'--out',str(out)]
    cfg=root/'experiments/agenticir'/('config_eval.yaml' if a.mode=='severity' else 'config_comp.yaml')
    proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':proc.pid,'start_time':start_time(proc.pid),'script':script,'gpu_uuid':a.gpu,'logical_cuda_device':0,'config_sha256':sha256(cfg),'host':'127.0.0.1','port':5001 if a.mode=='severity' else 5002,'ports':ports,'adapter_sha256':adapter_sha256}
    atomic_json(pidfile,record)
    for _ in range(180):
        if proc.poll() is not None:raise RuntimeError('service startup exited '+str(proc.returncode))
        with socket.socket() as s:
            s.settimeout(1)
            if all(_port_open(port) for port in ports):
                atomic_json(out/(a.mode+'_listening.json'),record);print('service ready for route validation',record['port'],proc.pid,flush=True);return
        time.sleep(1)
    os.killpg(proc.pid,signal.SIGTERM);raise TimeoutError('local service readiness budget exceeded')
if __name__=='__main__':main()
