"""Normalize a source-window increment; never decode its sealed test during preparation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from .prepare_vlm_data import prepare
from .manifest_io import write_csv, sha256
from .vlm_training import save_json


def prepare_increment(pack, out, scope):
    pack=Path(pack).resolve(); out=Path(out).resolve()
    if out.exists(): raise ValueError('increment preparation requires a new directory')
    ready=json.loads((pack/'READY.json').read_text())
    for f in ready['files']:
        # Byte verification is allowed for sealed test; no decoding or label construction.
        p=(pack/f['path']).resolve()
        if not p.is_relative_to(pack): raise ValueError('path outside source package')
        if sha256(p)!=f['sha256'] or p.stat().st_size!=f['bytes']: raise ValueError('source package integrity failure')
    rows=list(csv.DictReader((pack/'manifest.csv').open(encoding='utf-8-sig')))
    mappings=list(csv.DictReader((pack/'group_mapping.csv').open(encoding='utf-8-sig')))
    bygroup={r['group_id']:r for r in mappings}
    groups={split:{r['group_id'] for r in rows if r['split']==split} for split in ['train','val','test']}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]): raise ValueError('increment split leakage')
    old=json.loads(Path(scope).read_text())
    # Read the explicit excluded old group set; preserve the full scope in the evidence.
    exclusions=set(old['excluded_all_previous_dev_test_groups'])
    if any(r['old_candidate_group_id'] in exclusions for r in mappings): raise ValueError('old development/test source reused')
    if any(r['old_group_overlap']!='False' or r['alias_same_day_overlap']!='False' for r in mappings): raise ValueError('source association not resolved')
    selected=[]
    for group in sorted(groups['train']):
        candidates=sorted([r for r in rows if r['group_id']==group],key=lambda r:(float(r['timestamp_seconds']),r['sample_id']))
        selected.extend([candidates[0]] if len(candidates)==1 else [candidates[0],candidates[-1]])
    selected.extend(r for r in rows if r['split']=='val')
    out.mkdir(parents=True);normalized=[]
    for r in selected:
        image=pack/r['input_path']
        if sha256(image)!=r['input_sha256']: raise ValueError('image hash mismatch')
        m=bygroup[r['group_id']]
        normalized.append(dict(sample_id=r['sample_id'],group_id=r['group_id'],split='dev',domain='street',
            scene_id=m['location'],image_path=str(image),role='base' if r['normality']=='relatively_normal_or_mild' else 'real',
            lighting=r['lighting'],visual_tags=r['visual_tags']))
    norm=out/'normalized_pack';norm.mkdir();write_csv(norm/'manifest.csv',normalized)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),train_groups=sorted(groups['train']),
        val_groups=sorted(groups['val']),test_groups_excluded=sorted(groups['test']),
        sampling='At most earliest and latest valid PTS per new train source group; all validation sources; no visual or score-based selection',
        independent_train_images=sum(r['split']=='train' for r in selected),independent_val_images=sum(r['split']=='val' for r in selected),
        train_images_reserved=sum(r['split']=='train' for r in rows)-sum(r['split']=='train' for r in selected),
        epochs=3,lr=1e-5,accumulation=4,seed=929,initialization='unchanged author delta; not warm-started from prior experiments',
        selection='same validation macro four-class accuracy; tie earlier epoch; no final test access before selection',
        test_protocol='Same controlled impairments, identity and metadata-backed noncorrespondence; baseline versus selected fresh model; no test-based changes',
        attention='already-audited aligned batch-one wrapper',
        label_limitations='AI scene tags are not human restoration preferences; controlled perturbations are not MOS; location difference is only a prompt-specific correspondence control')
    save_json(out/'protocol.json',protocol)
    write_csv(out/'selected_sources.csv',selected)
    cams={s:{bygroup[g]['camera_full_code'] for g in groups[s] if bygroup[g]['camera_full_code']} for s in ['train','val']}
    save_json(out/'source_audit.json',dict(files_verified=len(ready['files']),ready_sha256=sha256(pack/'READY.json'),
        scope_sha256=sha256(scope),manifest_sha256=sha256(pack/'manifest.csv'),
        train_val_shared_camera_codes=sorted(cams['train']&cams['val']),no_unseen_camera_claim=True,
        test_decoded=False,excluded_old_groups_verified=True))
    prepare(norm,out/'protocol.json',out/'data')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',required=True);p.add_argument('--out',required=True);p.add_argument('--scope',required=True)
    a=p.parse_args();prepare_increment(a.pack,a.out,a.scope)
