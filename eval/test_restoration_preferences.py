import unittest
from .prepare_restoration_preferences import consensus, noncorresponding_sources

class ProxyConsensusTests(unittest.TestCase):
    def test_tradeoffs_and_near_ties_are_not_labels(self):
        margins={'psnr':.3,'ssim':.002,'lpips':.002}
        base={'psnr':30.,'ssim':.90,'lpips':.1}
        self.assertIsNone(consensus(base,base,margins))
        self.assertIsNone(consensus(dict(base,psnr=40.),base,margins))
        better={'psnr':31.,'ssim':.91,'lpips':.09}
        self.assertEqual(consensus(better,base,margins),'left')
        self.assertEqual(consensus(base,better,margins),'right')

    def test_missing_scene_is_not_evidence_of_noncorrespondence(self):
        a=dict(group_id='a',camera_id='camera_a',scene_id='')
        b=dict(group_id='b',camera_id='',scene_id='named_location')
        self.assertFalse(noncorresponding_sources(a,b))
        self.assertFalse(noncorresponding_sources(dict(a,scene_id='  '),b))
        self.assertTrue(noncorresponding_sources(a,dict(b,camera_id='camera_b')))
        self.assertFalse(noncorresponding_sources(dict(a,scene_id='other'),dict(b,camera_id='camera_a')))
        self.assertTrue(noncorresponding_sources(dict(a,scene_id='other'),b))

if __name__=='__main__':unittest.main()
