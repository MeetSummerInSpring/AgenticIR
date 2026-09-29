"""Unlock the new source test only after a completed, hash-identified model selection."""
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timezone
from .vlm_training import digest,save_json
from .manifest_io import write_csv
from .prepare_vlm_data import prepare


def unlock(pack,run,out):
    pack=Path(pack).resolve();run=Path(run).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('test output must be new')
    result=json.loads((run/'result.json').read_text())
    config=json.loads((run/'training_config.json').read_text())
    if result['status']!='completed' or result.get('study_scope')!='new_record_group_study':
        raise ValueError('new source model has not finished selection')
    if digest(run/'best_adapter.pt')!=result['best_adapter_sha256']:raise ValueError('selected checkpoint changed')
    if digest(config['data'])!=config['train_sha256'] or digest(config['val'])!=config['val_sha256']:raise ValueError('selection inputs changed')
    out.mkdir(parents=True)
    save_json(out/'selection_frozen_before_test_decode.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        selected_adapter=str(run/'best_adapter.pt'),selected_adapter_sha256=result['best_adapter_sha256'],
        best_epoch=result['best_epoch'],run_result_sha256=digest(run/'result.json'),
        no_reselection=True,comparison='untrained author delta versus this adapter; no test-based prompt/threshold or crop changes'))
    rows=list(csv.DictReader((pack/'manifest.csv').open(encoding='utf-8-sig')))
    maps={r['group_id']:r for r in csv.DictReader((pack/'group_mapping.csv').open(encoding='utf-8-sig'))}
    selected=[r for r in rows if r['split']=='test'];norm=[]
    train_val={r['group_id'] for r in rows if r['split'] in ['train','val']}
    if {r['group_id'] for r in selected}&train_val:raise ValueError('source split leakage')
    for r in selected:
        image=(pack/r['input_path']).resolve()
        if not image.is_relative_to(pack/'final_test'):raise ValueError('unexpected test path')
        if digest(image)!=r['input_sha256']:raise ValueError('test image changed')
        norm.append(dict(sample_id=r['sample_id'],group_id=r['group_id'],split='dev',domain='street',
            scene_id=maps[r['group_id']]['location'],image_path=str(image),role='base' if r['normality']=='relatively_normal_or_mild' else 'real',
            lighting=r['lighting'],visual_tags=r['visual_tags']))
    normalized=out/'normalized_pack';normalized.mkdir();write_csv(normalized/'manifest.csv',norm)
    protocol=dict(train_groups=[],val_groups=sorted({r['group_id'] for r in norm}),test_groups_excluded=[],
        scope='sealed new-source final diagnostic; technical val key only for shared constructor; NEVER train or select',
        source_split='test',model_selection_forbidden=True)
    save_json(out/'protocol.json',protocol)
    prepare(normalized,out/'protocol.json',out/'data')
    path=out/'data/val.json';items=json.loads(path.read_text())
    for item in items:item['split']='final_test'
    save_json(out/'test.json',items)
    save_json(path,items)  # Reject accidental training even if the constructor filename is used.
    pairs=json.loads((out/'data/pairs.json').read_text())
    for pair in pairs:pair['split']='final_test'
    save_json(out/'data/pairs.json',pairs)
    save_json(out/'audit.json',dict(n_sources=len(selected),n_groups=len({r['group_id'] for r in selected}),n_qa=len(items),
        source_split='test',constructor_internal_val_json='construction artifact only; use test.json for evaluation',
        independent_new_record_groups=True,unseen_camera_claim=False,human_preference_labels=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();unlock(a.pack,a.run,a.out)
