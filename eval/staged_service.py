"""On-demand local services for genuine single-GPU execution, never cached answers."""
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse
from .experiment_support import atomic_json, resources


class StagedServices:
    def __init__(self, out, gpu, max_starts=12):
        self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.gpu=gpu;self.max_starts=max_starts;self.starts=0;self.mode=None;self.events=[]

    def _record(self, event, **extra):
        self.events.append(dict(event=event,time=time.time(),active_mode=self.mode,**extra))
        atomic_json(self.out/'switches.json',self.events)

    def _command(self, action, mode):
        return [sys.executable,'-m','eval.local_service',action,'--mode',mode,'--out',str(self.out),'--gpu',self.gpu]

    def ensure_for_url(self, url):
        parsed=urlparse(url)
        if parsed.hostname!='127.0.0.1' or parsed.port not in [5001,5002]:raise ValueError('staged vision only supports existing local routes')
        mode='severity' if parsed.port==5001 else 'compare'
        if self.mode==mode:return
        self.release()
        if self.starts>=self.max_starts:raise RuntimeError('local model loading budget exhausted')
        self.starts+=1;start=time.monotonic()
        # Ownership record is written by local_service before model loading.
        self.mode=mode
        try:
            subprocess.run(self._command('start',mode),check=True,timeout=240,capture_output=True,text=True)
        except BaseException:
            try:self.release()
            except Exception as cleanup_error:self._record('cleanup_failed',error=str(cleanup_error))
            raise
        self._record('service_started',mode=mode,load_seconds=time.monotonic()-start)

    def release(self):
        if self.mode is None:return
        mode=self.mode;start=time.monotonic()
        if (self.out/(mode+'_owned_process.json')).exists():
            subprocess.run(self._command('stop',mode),check=True,timeout=30,capture_output=True,text=True)
        self.mode=None
        for _ in range(30):
            state=resources()
            if not any(self.gpu in line for line in state['processes']['stdout'].splitlines()):
                self._record('service_stopped',mode=mode,unload_seconds=time.monotonic()-start);return
            time.sleep(.5)
        raise RuntimeError('GPU still occupied after owned-service stop; refusing tool execution')


def attach_staging(agent, controller, catalog=None):
    """Attach deployment hooks to an existing ExperimentAgent without changing its policy."""
    post=agent.depictqa.session.post
    def staged_post(url,*args,**kwargs):
        controller.ensure_for_url(url)
        return post(url,*args,**kwargs)
    agent.depictqa.session.post=staged_post
    class StagedTool:
        def __init__(self, tool):self.tool=tool;self.tool_name=tool.tool_name
        def __call__(self,*args,**kwargs):
            controller.release()
            return self.tool(*args,**kwargs)
    router={}
    for task,tools in agent.executor.toolbox_router.items():
        if catalog is not None:
            if task not in catalog:raise ValueError('catalog must explicitly cover every registered task')
            wanted=catalog[task];known={t.tool_name for t in tools}
            if not wanted or not set(wanted)<=known:raise ValueError('invalid catalog tool selection: '+task)
            tools=[next(t for t in tools if t.tool_name==name) for name in wanted]
        router[task]=[StagedTool(t) for t in tools]
    agent.executor.toolbox_router=router
    agent._append_episode_event(event_type='deployment_configuration',state=agent._episode_state(),outcome={'mode':'single_gpu_live_staged','catalog':{k:[t.tool_name for t in v] for k,v in router.items()},'response_cache':False})


def attach_catalog(agent, catalog):
    """Select the same explicit available tools for persistent-service experiments."""
    router={}
    if set(catalog) != set(agent.executor.toolbox_router):
        raise ValueError('catalog must explicitly cover every registered task')
    for task, tools in agent.executor.toolbox_router.items():
        wanted=catalog[task]; known={t.tool_name:t for t in tools}
        if not wanted or len(wanted)!=len(set(wanted)) or not set(wanted)<=set(known):
            raise ValueError('invalid catalog tool selection: '+task)
        router[task]=[known[name] for name in wanted]
    agent.executor.toolbox_router=router
    agent._append_episode_event(event_type='deployment_configuration',state=agent._episode_state(),
        outcome={'mode':'persistent_local_services','catalog':catalog,'response_cache':False})
