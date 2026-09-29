import copy
import tempfile
from pathlib import Path
import unittest
from .vlm_training import canonical_prediction,verify_rows,digest,align_single_conversation


class TrainingGuards(unittest.TestCase):
    def test_prediction_parser_is_shared_and_not_substring_matching(self):
        for text,wanted in [('A','A'),('Image B.','B'),('Tie','Tie'),('Uncertain.','Uncertain'),('Both images differ','invalid'),('Astonishing detail','invalid')]:
            self.assertEqual(canonical_prediction(text),wanted)

    def make_rows(self,root):
        paths=[]
        for i in range(4):
            p=root/str(i);p.write_text(str(i));paths.append(str(p))
        def row(split,group,a,b):return dict(split=split,group_id=group,image_A=a,image_B=b,image_A_sha256=digest(a),image_B_sha256=digest(b))
        return [row('train','g1',*paths[:2])],[row('val','g2',*paths[2:])]

    def test_valid_independent_groups_and_images(self):
        with tempfile.TemporaryDirectory() as d:verify_rows(*self.make_rows(Path(d)))

    def test_secondary_pair_group_leak_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            train,val=self.make_rows(Path(d));train[0]['other_group_id']='g2'
            with self.assertRaisesRegex(ValueError,'source groups'):verify_rows(train,val)

    def test_duplicate_image_rejected_even_if_group_renamed(self):
        with tempfile.TemporaryDirectory() as d:
            train,val=self.make_rows(Path(d));val[0]['image_A']=train[0]['image_A'];val[0]['image_A_sha256']=train[0]['image_A_sha256']
            with self.assertRaisesRegex(ValueError,'duplicate'):verify_rows(train,val)

    def test_changed_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            train,val=self.make_rows(Path(d));Path(train[0]['image_B']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'input changed'):verify_rows(train,val)

    def test_test_or_other_domain_cannot_train(self):
        with tempfile.TemporaryDirectory() as d:
            train,val=self.make_rows(Path(d));train[0]['split']='test'
            with self.assertRaises(ValueError):verify_rows(train,val)
            train[0]['split']='train';train[0]['domain']='tianlian'
            with self.assertRaises(ValueError):verify_rows(train,val)


class AttentionParity(unittest.TestCase):
    def test_real_separator_remains_visible_and_batch_padding_is_rejected(self):
        import torch
        ids=torch.tensor([[1,2,3]])
        targets=torch.tensor([[-100,-100,3]])
        old_mask=torch.tensor([[1,0,1]])
        tokenize=align_single_conversation(lambda c,t:(ids,targets,old_mask))
        actual_ids,actual_targets,actual_mask=tokenize([[{}]],'quality_compare_noref')
        self.assertTrue(torch.equal(actual_mask,torch.ones_like(old_mask)))
        self.assertIs(actual_ids,ids)
        self.assertIs(actual_targets,targets)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            tokenize([[{}],[{}]],'quality_compare_noref')


class FrozenGateGuards(unittest.TestCase):
    def test_requires_both_orderings_and_abstains_on_invalid_or_tie(self):
        from .vlm_validation_suite import frozen_gate_decisions
        for forward, reverse, expected in [('B','A',True),('A','B',False),
                                           ('B','B',False),('Tie','Tie',False),
                                           ('invalid','A',False),('Uncertain','A',False)]:
            records=[dict(pair_id='p',order='forward',prediction=forward),
                     dict(pair_id='p',order='reverse',prediction=reverse)]
            self.assertEqual(frozen_gate_decisions(records)[0]['accept_fixed_right_candidate'],expected)
        self.assertFalse(frozen_gate_decisions([dict(pair_id='p',order='forward',prediction='B')])[0]['accept_fixed_right_candidate'])


class SummaryGuards(unittest.TestCase):
    def test_unlabeled_reviews_never_become_accuracy(self):
        from .vlm_report import summarize
        rows=[dict(id='1',pair_id='p',order='forward',prediction='B',answer='unlabeled',group_id='g'),
              dict(id='2',pair_id='p',order='reverse',prediction='A',answer='unlabeled',group_id='g')]
        result=summarize(rows)
        self.assertIsNone(result['accuracy'])
        self.assertIsNone(result['macro_accuracy'])
        self.assertEqual(result['n_labeled'],0)
        self.assertEqual(result['swap_consistency'],1.)

    def test_invalid_pair_is_not_consistent_and_macro_weights_classes(self):
        from .vlm_report import summarize
        rows=[dict(pair_id='p',order='forward',prediction='invalid',answer='Tie',group_id='g'),
              dict(pair_id='p',order='reverse',prediction='invalid',answer='Tie',group_id='g'),
              dict(pair_id='q',order='forward',prediction='A',answer='A',group_id='g')]
        result=summarize(rows)
        self.assertEqual(result['accuracy'],1/3)
        self.assertEqual(result['macro_accuracy'],.5)
        self.assertEqual(result['swap_consistency'],0.)


class SealedTestGuards(unittest.TestCase):
    def test_no_test_preparation_without_completed_fresh_selection(self):
        import json
        from .prepare_vlm_final_test import unlock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();out=root/'out'
            (run/'result.json').write_text(json.dumps(dict(status='completed',study_scope='old_development_sources')))
            (run/'training_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'not finished selection'):
                unlock(root/'pack',run,out)
            self.assertFalse(out.exists())

    def test_checkpoint_mismatch_rejected_before_accessing_test(self):
        import json
        from .prepare_vlm_final_test import unlock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();out=root/'out'
            (run/'result.json').write_text(json.dumps(dict(status='completed',study_scope='new_record_group_study',best_adapter_sha256='wrong')))
            (run/'training_config.json').write_text('{}');(run/'best_adapter.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                unlock(root/'pack',run,out)
            self.assertFalse(out.exists())
