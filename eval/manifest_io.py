"""Strict experiment manifests; relative paths are relative to the CSV."""
import csv
import hashlib
import os
import tempfile
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    # Commit a complete CSV atomically so interrupted batches retain a valid checkpoint.
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        temporary = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def resolve(path, root):
    return str((Path(root) / path).resolve()) if path else ''


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    rows = read_csv(path); seen = set(); groups = {}
    for r in rows:
        key = r['sample_id']
        if key in seen: raise ValueError('duplicate sample_id: ' + key)
        seen.add(key)
        group = (r['domain'], r['group_id'])
        if group in groups and groups[group] != r['split']:
            raise ValueError('split conflict: ' + str(group))
        groups[group] = r['split']
        r['input_path'] = resolve(r.get('input_path') or r.get('image_path'), Path(path).parent)
        r['reference_path'] = resolve(r.get('reference_path', ''), Path(path).parent)
        if r['role'] == 'real' and r['reference_path']:
            raise ValueError('real cannot have a full-reference target: ' + key)
        if r['role'] == 'synthetic' and not r['reference_path']:
            raise ValueError('synthetic reference missing: ' + key)
    return rows
