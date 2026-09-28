"""Tests for local tool integration and conservative acceptance diagnostics."""
import unittest
import numpy as np
from .real_acceptance import decide, preservation


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.valid={'roi_ssim':1.,'roi_gradient_cosine':1.,'new_clipped_fraction':0.,'mean_absolute_rgb_change':0.}

    def test_disagreement_keeps_original(self):
        self.assertFalse(decide([True,False],self.valid)[0])
        self.assertFalse(decide([True],self.valid)[0])
        self.assertFalse(decide([],self.valid)[0])

    def test_content_gate_independent_of_preference(self):
        self.assertFalse(decide([True,True],dict(self.valid,roi_ssim=.7))[0])
        self.assertFalse(decide([True,True],dict(self.valid,roi_gradient_cosine=.3))[0])
        self.assertFalse(decide([True,True],dict(self.valid,new_clipped_fraction=.01))[0])
        self.assertFalse(decide([True,True],dict(self.valid,roi_ssim=float('nan')))[0])
        self.assertTrue(decide([True,True],self.valid)[0])

    def test_real_content_perturbation(self):
        x=np.random.RandomState(1).rand(64,64,3).astype(np.float32)
        same=preservation(x,x,[0,0,1,1]);self.assertAlmostEqual(same['roi_ssim'],1)
        self.assertAlmostEqual(same['roi_gradient_cosine'],1,places=5)
        changed=preservation(x,1-x,[0,0,1,1]);self.assertFalse(decide([True,True],changed)[0])
        with self.assertRaises(ValueError):preservation(x,x[:32],[0,0,1,1])


class WrapperTests(unittest.TestCase):
    def test_registration_and_standard_contract(self):
        from executor import executor
        from executor.tool import Tool
        candidates=[t for t in executor.toolbox_router['deraining'] if t.tool_name=='histoformer_real']
        self.assertEqual(len(candidates),1);self.assertIsInstance(candidates[0],Tool)
        self.assertTrue(candidates[0].config_path.is_file())
