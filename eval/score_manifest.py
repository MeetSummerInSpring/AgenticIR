"""Manifest scoring without implicit pairing, resizing, or metric downloads.
python -m eval.score_manifest --manifest ... --results ... --out ...
"""
import argparse
import collections
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from .manifest_io import load_manifest, read_csv, write_csv, resolve, sha256

DIRECTIONS = {'psnr': 'higher', 'ssim': 'higher', 'lpips': 'lower', 'dists': 'lower', 'niqe': 'lower', 'musiq': 'higher'}
FR = {'psnr', 'ssim', 'lpips', 'dists'}


def pixels(path):
    with Image.open(path) as im:
        im.load()
        if im.mode != 'RGB': raise ValueError('expected RGB 8-bit image, got ' + im.mode)
        x = np.asarray(im, dtype=np.float32) / 255.
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1: raise ValueError('invalid pixels')
    return x


class Metrics:
    def __init__(self, device='cpu', weights_dir=None):
        self.device = device; self.models = {}; self.errors = {}; self.weights_dir = Path(weights_dir) if weights_dir else None

    def score(self, name, x, y=None):
        if name in FR and (y is None or x.shape != y.shape): raise ValueError('reference missing or shape mismatch')
        if name == 'psnr':
            mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
            return float('inf') if mse == 0 else float(-10*np.log10(mse))
        if name == 'ssim':
            from skimage.metrics import structural_similarity
            return float(structural_similarity(x, y, data_range=1, channel_axis=2, win_size=11,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        if name in self.errors: raise RuntimeError(self.errors[name])
        if name not in self.models:
            # Pretrained libraries may download inside constructors. Forbid downloads;
            # existing cached weights work through the original loader's cache check.
            import torch.hub
            torch.set_num_threads(min(4, torch.get_num_threads()))
            try:
                import pyiqa
            except Exception as exc:
                self.errors[name] = type(exc).__name__ + ': ' + str(exc)
                raise RuntimeError(self.errors[name]) from exc
            original = torch.hub.download_url_to_file
            def offline(*args, **kwargs):
                raise FileNotFoundError('uncached pretrained weight; automatic download disabled')
            torch.hub.download_url_to_file = offline
            import pyiqa.utils.download_util as du
            original_du = getattr(du, 'download_url_to_file', None)
            if original_du: du.download_url_to_file = offline
            try:
                options = {}
                filenames = {'lpips':'LPIPS_v0.1_alex-df73285e.pth','dists':'DISTS_weights-f5e65c96.pth','niqe':'niqe_modelparameters.mat'}
                if self.weights_dir and name in filenames:
                    weight = self.weights_dir / filenames[name]
                    if weight.exists(): options['pretrained_model_path'] = str(weight)
                self.models[name] = pyiqa.create_metric(name, device=self.device, **options)
            except Exception as exc:
                self.errors[name] = type(exc).__name__ + ': ' + str(exc)
                raise RuntimeError(self.errors[name]) from exc
            finally:
                torch.hub.download_url_to_file = original
                if original_du: du.download_url_to_file = original_du
        import torch
        def tensor(a): return torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.models[name](tensor(x), tensor(y)) if y is not None else self.models[name](tensor(x))
        value = float(value.item())
        if not math.isfinite(value): raise ValueError('metric returned nonfinite value')
        return value


def evaluate(manifest, results, out, names, device='cpu', weights_dir=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(manifest); index = {r['sample_id']: r for r in samples}
    records = read_csv(results); lookup = {}; methods = set()
    for r in records:
        key = (r['sample_id'], r['method'])
        if key in lookup: raise ValueError('duplicate result ' + str(key))
        if key[0] not in index: raise ValueError('unknown sample_id ' + key[0])
        lookup[key] = r; methods.add(r['method'])
    if not methods: raise ValueError('results must declare at least one method')
    engine = Metrics(device, weights_dir); detail = []
    for s in samples:
        # Base images are references, not evaluation samples.
        if s['role'] == 'base': continue
        for method in sorted(methods):
            r = lookup.get((s['sample_id'], method), {})
            error = ''; x = y = z = None
            try:
                x = pixels(s['input_path'])
                for field in ('input', 'reference'):
                    expected = s.get(field+'_sha256')
                    if expected and sha256(s[field+'_path']) != expected: raise ValueError(field+' checksum mismatch')
                if s['reference_path']:
                    y = pixels(s['reference_path'])
                    if x.shape != y.shape: raise ValueError('input/reference shape mismatch')
                for field in ('input_path', 'reference_path'):
                    if not r or field not in r or resolve(r[field], Path(results).parent) != s[field]:
                        raise ValueError('pairing mismatch: ' + field)
                if not r.get('output_path'): raise ValueError(r.get('error') or 'missing output')
                z = pixels(resolve(r['output_path'], Path(results).parent))
                if z.shape != x.shape: raise ValueError('output/input shape mismatch')
            except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
            for name in names:
                row = {k: s.get(k, '') for k in ['sample_id','domain','role','split','group_id','base_id','condition']}
                row.update(method=method, metric=name, direction=DIRECTIONS[name], input_score='', output_score='', delta='', improvement='', status='failed', reason=error, output_validation='failed' if error else 'ok', output_error=error)
                if name in FR and y is None and not s['reference_path']:
                    row.update(status='no_reference', reason='real sample: full-reference metrics inapplicable')
                else:
                    try:
                        if x is not None and (name not in FR or y is not None): row['input_score'] = engine.score(name,x,y if name in FR else None)
                        if error: raise ValueError(error)
                        row['output_score'] = engine.score(name,z,y if name in FR else None)
                        delta = row['output_score'] - row['input_score']
                        if not math.isnan(delta):
                            row['delta'] = delta; row['improvement'] = delta if DIRECTIONS[name]=='higher' else -delta
                        row.update(status='ok', reason='undefined inf-inf delta' if math.isnan(delta) else '')
                    except Exception as exc: row['reason'] = str(exc)
                detail.append(row)
    write_csv(out/'metrics_per_sample.csv',detail)
    grouped = collections.defaultdict(list)
    dims = ['domain','role','split','condition','method','metric']
    for r in detail: grouped[tuple(r[k] for k in dims)].append(r)
    summary = []
    for key, rows in grouped.items():
        valid = [r for r in rows if r['status']=='ok']; equal = collections.defaultdict(list); base_equal = collections.defaultdict(list)
        for r in valid:
            equal[(r['domain'],r['group_id'])].append(r['output_score'])
            base_equal[(r['domain'],r['base_id'] or r['sample_id'])].append(r['output_score'])
        # Common set requires every declared method to have valid scores on that image.
        common = []
        for r in valid:
            peers = [q for q in detail if q['sample_id']==r['sample_id'] and q['metric']==r['metric']]
            if len(peers)==len(methods) and all(q['status']=='ok' for q in peers): common.append(r)
        mean = lambda a: float(np.mean(a)) if a else ''
        reasons = collections.Counter(r['reason'] for r in rows if r['status']!='ok')
        summary.append(dict(zip(dims,key), direction=DIRECTIONS[key[-1]], n_total=len(rows), n_valid=len(valid), n_common=len(common),
            n_source_groups=len(equal), n_bases=len(base_equal), base_equal_output_mean=mean([mean(v) for v in base_equal.values()]), output_mean=mean([r['output_score'] for r in valid]),
            input_mean=mean([r['input_score'] for r in valid]), delta_mean=mean([r['delta'] for r in valid if r['delta']!='']),
            source_equal_output_mean=mean([mean(v) for v in equal.values()]), common_output_mean=mean([r['output_score'] for r in common]),
            missing_reasons=json.dumps(reasons,ensure_ascii=False)))
    write_csv(out/'metrics_summary.csv',summary)
    from itertools import combinations
    pair_rows = []
    for left, right in combinations(sorted(methods), 2):
        sets = collections.defaultdict(dict)
        for r in detail:
            if r['method'] in (left,right) and r['status']=='ok':
                sets[(r['domain'],r['role'],r['split'],r['condition'],r['metric'])].setdefault(r['sample_id'],{})[r['method']] = r
        for key, members in sets.items():
            common = [v for v in members.values() if left in v and right in v]
            pair_rows.append(dict(zip(['domain','role','split','condition','metric'],key),method_a=left,method_b=right,n_common=len(common)))
    write_csv(out/'metrics_common_pairs.csv',pair_rows, ['domain','role','split','condition','metric','method_a','method_b','n_common'])

    versions = {}
    for pkg in ['numpy','Pillow','scikit-image','torch','torchvision','pyiqa']:
        try: versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: versions[pkg] = 'missing'
    (out/'metric_protocol.json').write_text(json.dumps({'RGB':True,'range':[0,1],'crop_border':0,'resize':False,
        'PSNR':'all-channel MSE float64, inf preserved', 'SSIM':{'implementation':'skimage','win_size':11,'gaussian_weights':True,'sigma':1.5,'use_sample_covariance':False,'data_range':1,'channel_axis':2},
        'LPIPS':'PyIQA Alex v0.1; [0,1] -> [-1,1]; ScalingLayer shift [-0.030,-0.088,-0.188], scale [0.458,0.448,0.450]; no spatial resize',
        'DISTS':'PyIQA VGG16 ImageNet normalization; no resize in installed implementation',
        'versions':versions,'device':device,'unavailable':engine.errors, 'weights':{p.name:sha256(p) for p in Path(weights_dir).glob('*') if p.is_file()} if weights_dir else {}, 'NIQE':'PyIQA original, YIQ luminance, crop_border=0; native block statistics and internal half-scale pyramid'},indent=2))
    return detail


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--results',required=True); p.add_argument('--out',required=True)
    p.add_argument('--metrics',default='psnr,ssim,lpips,dists,niqe'); p.add_argument('--device',default='cpu'); p.add_argument('--weights-dir'); a=p.parse_args()
    names=a.metrics.split(',')
    if set(names)-DIRECTIONS.keys(): p.error('unsupported metric')
    evaluate(a.manifest,a.results,a.out,names,a.device,a.weights_dir)

if __name__=='__main__': main()
