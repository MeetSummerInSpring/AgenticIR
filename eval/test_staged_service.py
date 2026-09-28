import tempfile
from pathlib import Path
from unittest.mock import patch,MagicMock
import unittest
from .staged_service import StagedServices,attach_staging


class StagedServiceTests(unittest.TestCase):
    def test_routes_switch_and_reuse(self):
        with tempfile.TemporaryDirectory() as d,patch('eval.staged_service.subprocess.run') as run,patch('eval.staged_service.resources',return_value={'processes':{'stdout':''}}):
            c=StagedServices(d,'GPU-test');Path(d,'severity_owned_process.json').touch();Path(d,'compare_owned_process.json').touch();c.ensure_for_url('http://127.0.0.1:5001/evaluate_degradation');c.ensure_for_url('http://127.0.0.1:5001/evaluate_degradation')
            self.assertEqual(c.starts,1);c.ensure_for_url('http://127.0.0.1:5002/compare_quality');self.assertEqual(c.starts,2);c.release()
            actions=[call.args[0][3] for call in run.call_args_list];self.assertEqual(actions,['start','stop','start','stop'])
            with self.assertRaises(ValueError):c.ensure_for_url('https://example.com/compare')

    def test_budget_and_failed_start_cleanup(self):
        with tempfile.TemporaryDirectory() as d,patch('eval.staged_service.subprocess.run') as run,patch('eval.staged_service.resources',return_value={'processes':{'stdout':''}}):
            c=StagedServices(d,'GPU-test',max_starts=0)
            with self.assertRaises(RuntimeError):c.ensure_for_url('http://127.0.0.1:5001/x')
            self.assertEqual(run.call_count,0)
            c.max_starts=1;run.side_effect=[RuntimeError('load failed'),MagicMock()]
            with self.assertRaises(RuntimeError):c.ensure_for_url('http://127.0.0.1:5001/x')
            self.assertIsNone(c.mode);self.assertEqual(run.call_count,1)

    def test_unowned_start_failure_does_not_stop_existing_service(self):
        with tempfile.TemporaryDirectory() as d,patch('eval.staged_service.subprocess.run',side_effect=RuntimeError('port busy')) as run,patch('eval.staged_service.resources',return_value={'processes':{'stdout':''}}):
            c=StagedServices(d,'GPU-test')
            with self.assertRaisesRegex(RuntimeError,'port busy'):c.ensure_for_url('http://127.0.0.1:5002/x')
            self.assertEqual(run.call_count,1);self.assertIsNone(c.mode)

    def test_tool_releases_service_before_execution(self):
        from types import SimpleNamespace
        calls=[]
        class Tool:
            tool_name='test'
            def __call__(self,*args,**kwargs):calls.append('tool');return 'output'
        c=MagicMock();c.release.side_effect=lambda:calls.append('release')
        post=MagicMock(return_value='response')
        agent=SimpleNamespace(depictqa=SimpleNamespace(session=SimpleNamespace(post=post)),executor=SimpleNamespace(toolbox_router={'task':[Tool()]}),_append_episode_event=MagicMock(),_episode_state=lambda:{})
        attach_staging(agent,c,{'task':['test']})
        self.assertEqual(agent.executor.toolbox_router['task'][0](),'output');self.assertEqual(calls,['release','tool'])
        self.assertEqual(agent.depictqa.session.post('http://127.0.0.1:5001/x'),'response');c.ensure_for_url.assert_called_once()

class NewToolTests(unittest.TestCase):
    def test_udr_configs_and_contract(self):
        from executor import executor
        from executor.tool import Tool
        from executor.deraining.udr import UDRS2Former
        candidates=[t for t in executor.toolbox_router['deraining'] if t.tool_name.startswith('udr_')]
        self.assertEqual({t.tool_name for t in candidates},{'udr_raindrop_real','udr_agan'})
        self.assertTrue(all(isinstance(t,Tool) for t in candidates))
        with self.assertRaises(ValueError):UDRS2Former('unknown')
