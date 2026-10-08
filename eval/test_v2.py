import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from .text_planner import validate_text_payload,TextTransport,MODEL,ModelAccessError


class TextTests(unittest.TestCase):
    def payload(self,text='Choose a plan for medium rain.'):
        return {'model':MODEL,'messages':[{'role':'user','content':text}]}

    def test_text_guard(self):
        validate_text_payload(self.payload())
        for text in ['/root/private/image.png','T_I1651ab3356','data:image/png;base64,aaa']:
            with self.assertRaises(ValueError):validate_text_payload(self.payload(text))
        with self.assertRaises(ValueError):validate_text_payload(self.payload([{'type':'image_url','image_url':{'url':'test'}}]))

    def test_quota_stops_and_usage_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            t=TextTransport(Path(d)/'usage.jsonl')
            response=Mock(status_code=403);response.json.return_value={'error':{'code':'AllocationQuota.FreeTierOnly'}}
            with patch('eval.text_planner.requests.post',return_value=response) as post:
                with self.assertRaises(ModelAccessError):t({},self.payload())
                with self.assertRaises(ModelAccessError):t({},self.payload())
                self.assertEqual(post.call_count,1)
            self.assertEqual(json.loads((Path(d)/'usage.jsonl').read_text())['http_status'],403)

    def test_usage_and_bounded_rate_limit(self):
        with tempfile.TemporaryDirectory() as d:
            t=TextTransport(Path(d)/'usage.jsonl');bad=Mock(status_code=429);bad.json.return_value={}
            good=Mock(status_code=200);good.json.return_value={'usage':{'prompt_tokens':5,'completion_tokens':2,'completion_tokens_details':{'reasoning_tokens':1}}}
            with patch('eval.text_planner.requests.post',side_effect=[bad,good]),patch('eval.text_planner.time.sleep'):
                self.assertIs(t({},self.payload()),good)
            self.assertEqual(t.calls,2)
            rec=[json.loads(r) for r in (Path(d)/'usage.jsonl').read_text().splitlines()];self.assertEqual(rec[-1]['usage']['prompt_tokens'],5)


class CandidateFailureTests(unittest.TestCase):
    def make_agent(self,root,fail_all=False):
        from pipeline.iragent import IRAgent
        from executor import ToolExecutionError,ToolRunResult
        agent=object.__new__(IRAgent);agent.plan=['deraining'];agent.workflow_logger=Mock();agent.with_reflection=False;agent.with_rollback=False;agent.levels=['very low','low','medium','high','very high']
        parent={'img_path':str(root/'parent'/'0-img'/'input.png'),'state':{},'children':{'deraining':{'tools':{'good':{'state':{},'img_path':str(root/'good.png')}}}}}
        agent.cur_node=parent;agent.work_mem={'n_invocations':0};agent._episode_state=lambda:{};agent.events=[];agent._append_episode_event=lambda **kwargs:agent.events.append(kwargs)
        agent._record_tool_res=lambda *args:{'state':{}};agent._dump_summary=lambda:None;agent._render_img_tree=lambda:None;agent._img_nickname=lambda p:'image'
        class Bad:
            tool_name='bad'
            def __call__(self,**kwargs):raise ToolExecutionError('bad','deraining','execution','missing file')
        class Good:
            tool_name='good'
            def __call__(self,**kwargs):return ToolRunResult('good','deraining',.1,kwargs['output_dir']/'output.png')
        agent._prepare_for_subtask=lambda s:(root/'subtask','rain',[Bad()] if fail_all else [Bad(),Good()])
        agent._tool_transition=lambda *args:{}
        return agent

    def test_next_candidate_runs_after_execution_failure(self):
        with tempfile.TemporaryDirectory() as d:
            agent=self.make_agent(Path(d));self.assertTrue(agent.execute_subtask(None));self.assertEqual(agent.work_mem['n_invocations'],2)
            self.assertEqual(agent.events[0]['outcome']['status'],'execution_error');self.assertEqual(agent.events[1]['outcome']['status'],'ok')

    def test_all_failed_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            agent=self.make_agent(Path(d),True)
            with self.assertRaisesRegex(RuntimeError,'No valid result'):agent.execute_subtask(None)


class IsolationSwitchTests(unittest.TestCase):
    def test_snapshot_off_online_and_registry_are_independent(self):
        from types import SimpleNamespace
        from .experiment_agent import ExperimentAgent,IRAgent
        from utils.episode_store import EpisodeStore
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);snapshot=EpisodeStore(root/'snapshot.sqlite3')
            def add(store):store.append_event(run_id='dev',event_type='tool_attempt',action={'subtask':'deraining','tool':'one'},outcome={'status':'ok','quality_success':True})
            add(snapshot)
            def fake_init(agent,*args,**kwargs):
                agent.executor=SimpleNamespace(toolbox_router={'deraining':[SimpleNamespace(tool_name='two'),SimpleNamespace(tool_name='one')]})
                agent.episode_store=EpisodeStore(root/('events_'+str(id(agent))+'.sqlite3'))
                agent.tool_selector=SimpleNamespace(episode_store=agent.episode_store)
                agent.gpt4=SimpleNamespace(model=MODEL);agent.work_dir=root;agent.depictqa=None
                agent.with_retrieval=kwargs.get('with_retrieval',True)
            with patch.object(IRAgent,'__init__',fake_init):
                frozen=ExperimentAgent(memory_snapshot=root/'snapshot.sqlite3',with_retrieval=False)
                add(frozen.episode_store)
                self.assertFalse(frozen.with_retrieval)
                self.assertEqual(frozen.tool_selector.episode_store.get_tool_statistics('deraining')['one'].attempts,1)
                off=ExperimentAgent(memory_snapshot=root/'snapshot.sqlite3',memory_read='off')
                self.assertEqual(off.tool_selector.episode_store.get_tool_statistics('deraining'),{})
                online=ExperimentAgent(memory_update=True);add(online.episode_store)
                self.assertIs(online.tool_selector.episode_store,online.episode_store)
                self.assertEqual(online.tool_selector.episode_store.get_tool_statistics('deraining')['one'].attempts,1)
                registry=ExperimentAgent(selector_policy='registry')
                chosen,_=registry.tool_selector.select('deraining',registry.executor.toolbox_router['deraining'])
                self.assertEqual([t.tool_name for t in chosen],['two','one'])


class ScopeTests(unittest.TestCase):
    def test_scale_and_observed_state_gate(self):
        from .frozen_statistics import FrozenStatistics
        class Store:
            def iter_events(self,**kwargs):
                yield {'action':{'subtask':'deraining','tool':'one'},'state':{'scale_long_edge':1024,'observed_rain_severity':'medium'},'outcome':{'status':'ok','quality_success':True},'transition':{'duration_seconds':2}}
            def get_tool_statistics(self,subtask):return {'one':'evidence'}
        state={'scale_long_edge':512,'observed_rain_severity':'medium'}
        frozen=FrozenStatistics(Store(),context=lambda:state)
        self.assertEqual(frozen.get_tool_statistics('deraining'),{})
        state['scale_long_edge']=1024
        self.assertEqual(frozen.get_tool_statistics('deraining')['one'].quality_successes,1)
        # An additional scope must not contaminate the matching scope's estimates.
        class Mixed(Store):
            def iter_events(self,**kwargs):
                yield from super().iter_events(**kwargs)
                yield {'action':{'subtask':'deraining','tool':'one'},'state':{'scale_long_edge':512,'observed_rain_severity':'high'},'outcome':{'status':'ok','quality_success':False},'transition':{'duration_seconds':20}}
        mixed=FrozenStatistics(Mixed(),context=lambda:state)
        matched=mixed.get_tool_statistics('deraining')['one']
        self.assertEqual((matched.attempts,matched.quality_successes,matched.mean_duration_seconds),(1,1,2))
        state['observed_rain_severity']='high'
        self.assertEqual(frozen.get_tool_statistics('deraining'),{})

if __name__=='__main__':unittest.main()
