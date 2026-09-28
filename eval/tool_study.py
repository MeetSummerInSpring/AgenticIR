"""Sequential single-GPU tool evidence with immutable resume fingerprints."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from .manifest_io import load_manifest,read_csv,write_csv,sha256
from .experiment_support import atomic_json,resources,fingerprint,code_hashes,failure_category


def worker(spec):
    import random,numpy as np,torch
    from executor import executor
    from .score_manifest import pixels
    random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed']);torch.backends.cudnn.benchmark=False
    work=Path(spec['work']);inp=work/'input';inp.mkdir(parents=True,exist_ok=True);shutil.copy2(spec['input_path'],inp/'input.png')
    durations=[];current=inp
    for i,name in enumerate(spec['tools']):
        tool=next(t for t in executor.toolbox_router[spec.get('subtask','deraining')] if t.tool_name==name)
        out=work/f'stage_{i}';out.mkdir();res=tool(current,out,silent=True);durations.append(res.duration_seconds);current=out
    output=res.output_path
    if pixels(output).shape!=pixels(spec['input_path']).shape:raise ValueError('output/input shape mismatch')
    atomic_json(work/'worker_result.json',{'output_path':str(output),'output_sha256':sha256(output),'tool_durations':durations})


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--out');p.add_argument('--tools',nargs='+',default=['restormer','mprnet']);p.add_argument('--limit',type=int,default=16);p.add_argument('--timeout',type=int,default=300);p.add_argument('--seed',type=int,default=928);p.add_argument('--worker-json');p.add_argument('--retry-failed',action='store_true');p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--protocol');p.add_argument('--subtask',choices=['deraining','brightening'],default='deraining');a=p.parse_args()
    if a.worker_json:worker(json.loads(Path(a.worker_json).read_text()));return
    if not a.manifest or not a.out:p.error('--manifest and --out required')
    rows=load_manifest(a.manifest)
    if any(r['split']!=a.split for r in rows):raise ValueError('manifest split differs from explicitly requested split')
    protocol_hash=None
    if a.split=='test':
        if a.subtask!='deraining':raise ValueError('frozen holdout only supports deraining')
        if not a.protocol:raise ValueError('test execution requires a frozen protocol/manifest plan')
        from .prepare_holdout import validate_test_execution
        protocol_hash=validate_test_execution(a.manifest,a.tools,a.protocol)
        if a.limit<len(rows):raise ValueError('test scope must match frozen plan; resume uses checkpoints, not sample truncation')
    rows=rows[:a.limit];out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    resource=resources();atomic_json(out/('resource_'+str(time.time_ns())+'.json'),resource)
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError('this entry is the single allocated GPU study; select one allocated GPU explicitly')
    if resource['processes']['stdout']:raise RuntimeError('resource busy: other GPU processes present; do not stop them')
    inventory={}
    for name in a.tools:
        paths={'restormer':'executor/deraining/tools/Restormer/Deraining/pretrained_models/deraining.pth','mprnet':'executor/deraining/tools/MPRNet/Deraining/pretrained_models/model_deraining.pth','xrestormer':'executor/deraining/tools/X-Restormer/experiments/pretrained_models/XRestormer_deraining.pth','histoformer_real':'executor/deraining/tools/Histoformer/Allweather/pretrained_models/net_g_real.pth'}
        paths.update({'udr_raindrop_real':'executor/deraining/tools/UDR-S2Former/pretrained/udrs2former_raindrop_real.pth','udr_agan':'executor/deraining/tools/UDR-S2Former/pretrained/udrs2former_agan.pth','darkir_mt':'executor/brightening/tools/DarkIR/models/DarkIR_1k_cr_mt.pt'})
        if (name=='darkir_mt') != (a.subtask=='brightening'):raise ValueError('tool/subtask mismatch')
        if name not in paths:raise ValueError('unsupported study tool '+name)
        weight=Path(paths[name]);inventory[name]={'weight_path':str(weight),'weight_sha256':sha256(weight) if weight.exists() else 'missing'}
        folder=weight
        # Fingerprint the actual third-party code as well as its wrapper.
        root=Path('executor')/a.subtask/'tools'/({'restormer':'Restormer','mprnet':'MPRNet','xrestormer':'X-Restormer','histoformer_real':'Histoformer','udr_raindrop_real':'UDR-S2Former','udr_agan':'UDR-S2Former','darkir_mt':'DarkIR'}[name])
        inventory[name]['code']={str(f):sha256(f) for f in root.rglob('*.py') if '.git' not in f.parts}
    identity={'subtask':a.subtask,'split':a.split,'frozen_protocol_sha256':protocol_hash,'manifest':sha256(a.manifest),'inputs':{r['sample_id']:sha256(r['input_path']) for r in rows},'code':code_hashes(),'tools':inventory,
        'model':'none: fixed tools, no LLM','memory':'none','seed':a.seed,'timeout':a.timeout,'python':sys.version,'torch':torch.__version__,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_uuid':resource['gpus']['stdout'].split(',')[1].strip()}
    if a.split=='test':
        frozen=json.loads(Path(json.loads(Path(a.protocol).read_text())['frozen_protocol_path']).read_text())
        if any(inventory[n]['weight_sha256']!=frozen['weight_sha256'][n] for n in a.tools):raise ValueError('frozen tool weight changed')
    batch_id=fingerprint(identity);config=out/'run_identity.json'
    if config.exists() and json.loads(config.read_text())['fingerprint']!=batch_id:raise ValueError('resume identity changed: use new output version')
    atomic_json(config,{'fingerprint':batch_id,'identity':identity});write_csv(out/'manifest.csv',rows)
    records=read_csv(out/'results.csv') if (out/'results.csv').exists() else []
    for sample in rows:
        for name in a.tools:
            prior=next((r for r in records if r['sample_id']==sample['sample_id'] and r['method']==name),None)
            if prior and prior['status']=='ok' and Path(prior['output_path']).exists() and sha256(prior['output_path'])==prior['output_sha256']:continue
            if prior and prior['status']!='ok' and not a.retry_failed:continue
            row={k:sample.get(k,'') for k in ['sample_id','domain','role','split','group_id','base_id','condition','scale','input_path','reference_path']}
            row.update(method=name,mode='fixed_single_tool',status='failed',output_path='',error='',failure_category='',n_tool_calls=1,n_text_requests=0,n_local_evaluation_requests=0,batch_fingerprint=batch_id,input_sha256=sha256(sample['input_path']))
            work=out/'runs'/sample['sample_id']/name/str(time.time_ns());work.mkdir(parents=True);spec={'work':str(work),'input_path':sample['input_path'],'tools':[name],'seed':a.seed,'subtask':a.subtask};atomic_json(work/'spec.json',spec)
            t=time.monotonic();peak=0;proc=None
            try:
                # No VLM process shares this GPU during tool stage.
                current=resources()
                if current['processes']['stdout']:raise RuntimeError('resource busy: GPU process appeared before tool')
                env=os.environ.copy();env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
                with (work/'worker.log').open('w') as log:
                    proc=subprocess.Popen([sys.executable,'-m','eval.tool_study','--worker-json',str(work/'spec.json')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    while proc.poll() is None:
                        if time.monotonic()-t>a.timeout:
                            os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10);raise TimeoutError('per-tool budget exceeded')
                        usage=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                        try:peak=max(peak,int(usage.stdout.strip().splitlines()[0]))
                        except (ValueError,IndexError):pass
                        time.sleep(.5)
                if proc.returncode:raise RuntimeError((work/'worker.log').read_text()[-6000:])
                result=json.loads((work/'worker_result.json').read_text());row.update(result);row.update(status='ok',duration_tool_seconds=sum(result['tool_durations']))
            except Exception as exc:
                row['error']=str(exc);row['failure_category']=failure_category(exc)
                if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            row.update(duration_seconds=round(time.monotonic()-t,3),peak_device_memory_mib=peak,attempt_dir=str(work))
            records=[r for r in records if not(r['sample_id']==sample['sample_id'] and r['method']==name)]+[row]
            write_csv(out/'results.csv',records);print(sample['sample_id'],name,row['status'],row['duration_seconds'],peak,flush=True)
if __name__=='__main__':main()
