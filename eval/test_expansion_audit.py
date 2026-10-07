import unittest
from .audit_expansion import audit

class ExpansionAuditTests(unittest.TestCase):
    def row(self,sid,split,**kw):
        return dict(sample_id=sid,source_id=sid,group_id=sid,split=split,
                    prior_use='confirmed_project_unseen',camera_id='same_camera',**kw)
    def test_parent_and_reference_leakage(self):
        for field in ['parent_id','reference_sample_id']:
            rs=[self.row('a','train'),self.row('b','test',**{field:'a'})]
            self.assertFalse(audit(rs)['admissible'])
    def test_transitive_group_and_prior(self):
        prior=[self.row('a','dev')]
        r=self.row('b','test',parent_id='a')
        self.assertFalse(audit([r],prior)['admissible'])
    def test_camera_overlap_is_reported_not_source_leakage(self):
        result=audit([self.row('a','train'),self.row('b','test')])
        self.assertTrue(result['admissible'])
        self.assertEqual(result['camera_cross_split'],{'same_camera':['test','train']})
    def test_unknown_prior_not_new_final_test(self):
        r=self.row('a','test');r['prior_use']='competition_use_project_tuning_unknown'
        self.assertFalse(audit([r])['admissible'])

if __name__=='__main__':unittest.main()
