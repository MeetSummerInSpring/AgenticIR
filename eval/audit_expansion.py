"""Audit group/parent/reference lineage before constructing supervision.

This is an admission check, not a splitter. Unknown exposure is not eligible for
a new final test. Camera overlap is reported separately from source leakage.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from .manifest_io import read_csv
from .experiment_support import atomic_json


def audit(rows, prior=()):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def join(a, b):
        parent[find(a)] = find(b)
    errors = []
    seen = set()
    for r in rows:
        sid = r.get('sample_id', '')
        if not sid or sid in seen:
            errors.append({'sample_id': sid, 'reason': 'missing_or_duplicate_sample_id'})
        seen.add(sid)
    def tokens(r):
        values = [('sample', r.get('sample_id')), ('sample', r.get('parent_id')),
                  ('sample', r.get('reference_sample_id')),
                  ('group', r.get('group_id')), ('group', r.get('episode_id')),
                  ('source', r.get('source_id') or r.get('source_media_id')),
                  ('pixels', r.get('input_sha256')), ('pixels', r.get('reference_sha256'))]
        return [kind + ':' + str(value) for kind, value in values if value]
    for r in list(rows) + list(prior):
        ts = tokens(r)
        for t in ts[1:]:
            join(ts[0], t)
    components = defaultdict(list)
    for r in rows:
        ts = tokens(r)
        if ts:
            components[find(ts[0])].append(r)
    historical = {find(t) for r in prior for t in tokens(r)}
    for component, members in components.items():
        splits = {r.get('split') for r in members}
        if len(splits) > 1:
            errors.append({'reason': 'lineage_crosses_splits', 'samples': [r.get('sample_id') for r in members]})
        for r in members:
            if not r.get('group_id') or not r.get('source_id'):
                errors.append({'sample_id':r.get('sample_id'), 'reason':'source_and_group_required'})
            if r.get('split') == 'test' and (component in historical or r.get('prior_use') != 'confirmed_project_unseen'):
                errors.append({'sample_id':r.get('sample_id'), 'reason':'new_test_prior_exposure_or_unknown'})
    cameras = defaultdict(set)
    for r in rows:
        if r.get('camera_id'):
            cameras[r['camera_id']].add(r.get('split'))
    counts = {}
    for split in sorted({r.get('split', '') for r in rows}):
        rs = [r for r in rows if r.get('split', '') == split]
        counts[split] = {'rows':len(rs), 'sources':len({r.get('source_id') for r in rs if r.get('source_id')}),
                         'groups':len({r.get('group_id') for r in rs if r.get('group_id')}),
                         'cameras':len({r.get('camera_id') for r in rs if r.get('camera_id')}),
                         'asset_types':dict(Counter(r.get('asset_type','unknown') for r in rs))}
    return {'admissible':not errors, 'errors':errors, 'counts':counts,
            'camera_cross_split':{k:sorted(v) for k,v in cameras.items() if len(v)>1},
            'note':'Rows/derivatives are not independent original counts; aliases and weather episodes require source-side confirmation.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--prior',help='JSON metadata list or CSV');p.add_argument('--out',required=True);a=p.parse_args()
    prior=[]
    if a.prior:
        prior=json.loads(Path(a.prior).read_text()) if a.prior.endswith('.json') else read_csv(a.prior)
    result=audit(read_csv(a.manifest),prior);atomic_json(a.out,result)
    if not result['admissible']:
        raise SystemExit('data admission failed; inspect audit output')


if __name__=='__main__':main()
