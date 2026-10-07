import unittest
from .prepare_restoration_preferences import consensus

class ProxyConsensusTests(unittest.TestCase):
    def test_tradeoffs_and_near_ties_are_not_labels(self):
        margins={'psnr':.3,'ssim':.002,'lpips':.002}
        base={'psnr':30.,'ssim':.90,'lpips':.1}
        self.assertIsNone(consensus(base,base,margins))
        self.assertIsNone(consensus(dict(base,psnr=40.),base,margins))
        better={'psnr':31.,'ssim':.91,'lpips':.09}
        self.assertEqual(consensus(better,base,margins),'left')
        self.assertEqual(consensus(base,better,margins),'right')

if __name__=='__main__':unittest.main()
