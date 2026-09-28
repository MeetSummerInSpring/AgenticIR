import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from .manifest_io import write_csv, load_manifest
from .score_manifest import Metrics, evaluate, pixels


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.x=np.random.RandomState(4).rand(64,64,3).astype('float32')
        for name,x in [('input',self.x),('ref',self.x),('output',self.x*.8)]:
            Image.fromarray((x*255).astype('uint8')).save(self.root/(name+'.png'))
        self.s={'sample_id':'one','domain':'street','role':'synthetic','split':'dev','group_id':'g','condition':'rain','base_id':'b',
            'input_path':str(self.root/'input.png'),'reference_path':str(self.root/'ref.png')}
        self.r={'sample_id':'one','method':'a','input_path':self.s['input_path'],'reference_path':self.s['reference_path'],'output_path':str(self.root/'output.png')}

    def tearDown(self): self.tmp.cleanup()

    def score(self,samples=None,results=None):
        write_csv(self.root/'manifest.csv', samples or [self.s]); write_csv(self.root/'results.csv',results or [self.r])
        return evaluate(self.root/'manifest.csv',self.root/'results.csv',self.root/'scores',['psnr','ssim'])

    def test_identical_and_controlled_change(self):
        m=Metrics(); self.assertEqual(m.score('psnr',self.x,self.x),float('inf'))
        self.assertAlmostEqual(m.score('ssim',self.x,self.x),1,places=5)
        self.assertLess(m.score('ssim',self.x*.8,self.x),1)
        self.assertTrue(math.isfinite(m.score('psnr',self.x*.8,self.x)))

    def test_no_reference_real(self):
        self.s.update(role='real',reference_path=''); self.r['reference_path']=''
        self.assertTrue(all(r['status']=='no_reference' and r['output_score']=='' for r in self.score()))

    def test_missing_output(self):
        self.r['output_path']=''; rows=self.score()
        self.assertTrue(all(r['status']=='failed' and r['output_score']=='' for r in rows))
        self.assertEqual(rows[0]['input_score'],float('inf'))

    def test_wrong_explicit_pair(self):
        self.r['reference_path']=self.r['input_path']
        self.assertTrue(all('pairing mismatch' in r['reason'] for r in self.score()))

    def test_wrong_dimensions_no_resize(self):
        Image.new('RGB',(32,32)).save(self.root/'ref.png')
        self.assertTrue(all('shape mismatch' in r['reason'] for r in self.score()))

    def test_wrong_output_dimensions(self):
        Image.new('RGB',(32,32)).save(self.root/'output.png')
        self.assertTrue(all('shape mismatch' in r['reason'] for r in self.score()))

    def test_invalid_pixel_format(self):
        Image.new('L',(64,64)).save(self.root/'output.png')
        self.assertTrue(all('RGB' in r['reason'] for r in self.score()))

    def test_duplicate_and_split_conflict(self):
        with self.assertRaises(ValueError): self.score([self.s,self.s])
        other=dict(self.s,sample_id='two',split='test')
        with self.assertRaises(ValueError): self.score([self.s,other])

    def test_unknown_results_and_duplicates(self):
        with self.assertRaises(ValueError): self.score(results=[dict(self.r,sample_id='unknown')])
        with self.assertRaises(ValueError): self.score(results=[self.r,self.r])

    def test_infinite_delta_not_clipped(self):
        self.r['output_path']=self.s['input_path']; rows=self.score()
        self.assertEqual(rows[0]['output_score'],float('inf')); self.assertEqual(rows[0]['delta'],'')

    def test_common_valid_counts(self):
        self.score(results=[self.r,dict(self.r,method='b',output_path='')])
        from .manifest_io import read_csv
        self.assertTrue(all(r['n_common']=='0' for r in read_csv(self.root/'scores/metrics_summary.csv')))

    def test_frozen_statistics(self):
        from .frozen_statistics import FrozenStatistics
        class Store:
            x={'tool':'old'}
            def get_tool_statistics(self,subtask): return self.x
        store=Store(); frozen=FrozenStatistics(store); store.x['tool']='new'
        self.assertEqual(frozen.get_tool_statistics('deraining')['tool'],'old')
        self.assertEqual(FrozenStatistics().get_tool_statistics('deraining'),{})



@unittest.skipIf(__import__('sys').version_info < (3,10), 'runtime pipeline requires Python >=3.10')
class AdapterTests(unittest.TestCase):
    def test_budget_and_cold_memory(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .experiment_agent import ExperimentAgent, IRAgent
        class Tool:
            tool_name='one'
            def __call__(self): return 'ok'
        def fake_init(agent,*args,**kwargs):
            agent.executor=SimpleNamespace(toolbox_router={'deraining':[Tool()]})
            agent.gpt4=SimpleNamespace(model='deepseek-v4-flash-0731')
            agent.work_dir=Path(tempfile.gettempdir())/'agenticir_test_transport'
            agent.depictqa=None
            agent.tool_selector=SimpleNamespace(episode_store=None)
        class FakeTransport:
            def __init__(self,*args,**kwargs):self.calls=0;self.maximum=kwargs['max_calls']
            def __call__(self,*args):
                if self.calls>=self.maximum:raise RuntimeError('budget')
                self.calls+=1;return 'response'
        with patch.object(IRAgent,'__init__',fake_init), patch('eval.experiment_agent.TextTransport',FakeTransport):
            agent=ExperimentAgent(max_tool_calls=1,max_llm_calls=1)
            tool=agent.executor.toolbox_router['deraining'][0]
            self.assertEqual(tool(),'ok')
            with self.assertRaises(RuntimeError): tool()
            self.assertEqual(agent.gpt4._send_request({},{}),'response')
            with self.assertRaises(RuntimeError): agent.gpt4._send_request({},{})
            self.assertEqual(agent.tool_selector.episode_store.get_tool_statistics('deraining'),{})

if __name__=='__main__': unittest.main()
