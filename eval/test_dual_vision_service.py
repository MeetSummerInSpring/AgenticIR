import unittest
import threading
import time
import torch
from .dual_vision_service import DeltaRouter


class DeltaRouterTests(unittest.TestCase):
    def model(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2));self.abstractor=torch.nn.Linear(2,2,bias=False);self.calls=[]
            def generate(self,request):
                start=(self.weight.detach().clone(),self.abstractor.weight.detach().clone());time.sleep(.01)
                self.calls.append(request)
                assert torch.equal(start[0],self.weight) and torch.equal(start[1],self.abstractor.weight)
                return float(self.weight.sum()+self.abstractor.weight.sum())
        return Toy()
    def banks(self,model):
        return {mode:{n:torch.full_like(p,value) for n,p in model.named_parameters()} for mode,value in [('severity',1.),('compare',3.)]}
    def test_complete_roundtrip_and_repeated_real_generation(self):
        m=self.model();r=DeltaRouter(m,self.banks(m))
        self.assertEqual([r.generate(x,{}) for x in ['severity','compare','severity','severity']],[6.,18.,6.,6.]);self.assertEqual(len(m.calls),4);self.assertEqual(r.switches,3)
    def test_missing_abstractor_or_shape_rejected(self):
        m=self.model();b=self.banks(m);b['compare'].pop('abstractor.weight')
        with self.assertRaises(ValueError):DeltaRouter(m,b)
        b=self.banks(m);b['severity']['weight']=torch.ones(3)
        with self.assertRaises(ValueError):DeltaRouter(m,b)
    def test_concurrent_branches_cannot_mutate_generation(self):
        m=self.model();r=DeltaRouter(m,self.banks(m));answers={}
        threads=[threading.Thread(target=lambda mode:answers.update({mode:r.generate(mode,{})}),args=(mode,)) for mode in ['severity','compare']]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(answers,{'severity':6.,'compare':18.})
