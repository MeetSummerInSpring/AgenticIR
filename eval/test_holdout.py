import json
from pathlib import Path
import tempfile
import unittest
from .manifest_io import sha256,write_csv
from .prepare_holdout import validate_test_execution


class HoldoutGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.image=self.root/'input.png';self.image.write_bytes(b'controlled fixture')
        self.manifest=self.root/'manifest.csv';self.rows=[dict(sample_id='heldout',domain='a',role='real',split='test',group_id='g',input_path=str(self.image),input_sha256=sha256(self.image),reference_path='')];write_csv(self.manifest,self.rows)
        self.protocol=self.root/'frozen.json';self.protocol.write_text('{}')
        self.plan=self.root/'plan.json';self.reset_plan()

    def reset_plan(self):
        self.plan.write_text(json.dumps({'frozen_protocol_path':str(self.protocol),'frozen_protocol_sha256':sha256(self.protocol),'manifests':[{'sha256':sha256(self.manifest),'n':1,'tools':['restormer']}]}))

    def tearDown(self):self.tmp.cleanup()
    def test_exact_frozen_scope_passes(self):self.assertEqual(validate_test_execution(self.manifest,['restormer'],self.plan),sha256(self.protocol))
    def test_changed_tools_rejected(self):
        with self.assertRaises(ValueError):validate_test_execution(self.manifest,['mprnet'],self.plan)
    def test_modified_input_rejected(self):
        self.image.write_bytes(b'changed')
        with self.assertRaises(ValueError):validate_test_execution(self.manifest,['restormer'],self.plan)
    def test_relabelled_development_rejected(self):
        self.rows[0]['split']='dev';write_csv(self.manifest,self.rows);self.reset_plan()
        with self.assertRaises(ValueError):validate_test_execution(self.manifest,['restormer'],self.plan)
    def test_protocol_change_rejected(self):
        self.protocol.write_text('{"changed":true}')
        with self.assertRaises(ValueError):validate_test_execution(self.manifest,['restormer'],self.plan)
    def test_selection_change_rejected(self):
        self.rows[0]['sample_id']='different';write_csv(self.manifest,self.rows)
        with self.assertRaises(ValueError):validate_test_execution(self.manifest,['restormer'],self.plan)
