"""Experimental terminal acceptance over the selected trajectory, without labels.

Runs after search/rollback completes; it cannot save already incurred tool calls.
Both native binary orders must prefer a challenger. Disagreement is unresolved,
not a learned Tie/Uncertain prediction. No image similarity threshold is used.
"""
import hashlib
from pathlib import Path


def trajectory_prefix(tree, final_path):
    if str(tree['img_path']) == str(final_path):
        return [tree]
    for branch in tree.get('children', {}).values():
        for node in branch.get('tools', {}).values():
            found = trajectory_prefix(node, final_path)
            if found:
                return [tree] + found
    return []


def pixel_identity(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('RGB')
        return im.size, hashlib.sha256(im.tobytes()).hexdigest()


def accept_prefix(nodes, compare, identity=pixel_identity):
    if not nodes:
        raise ValueError('selected final image is absent from the execution tree')
    incumbent = nodes[0]
    seen = {identity(incumbent['img_path'])}
    decisions = []
    for challenger in nodes[1:]:
        path = challenger['img_path']
        digest = identity(path)
        item = {'incumbent': incumbent['img_path'], 'challenger': path}
        if digest in seen:
            item.update(outcome='duplicate_pixels', comparator_calls=0)
        else:
            ab = compare(Path(incumbent['img_path']), Path(path))
            ba = compare(Path(path), Path(incumbent['img_path']))
            if ab not in {'former', 'latter'} or ba not in {'former', 'latter'}:
                raise ValueError('native comparison must return former or latter')
            accept = ab == 'latter' and ba == 'former'
            item.update(ab=ab, ba=ba, comparator_calls=2,
                        outcome='accept_challenger' if accept else
                        'retain_incumbent' if ab == 'former' and ba == 'latter' else 'order_disagreement')
            if accept:
                incumbent = challenger
            seen.add(digest)
        item['selected'] = incumbent['img_path']
        decisions.append(item)
    return incumbent, {'policy': 'prefix_bidirectional_v1',
                       'original_final': nodes[-1]['img_path'],
                       'selected': incumbent['img_path'], 'decisions': decisions,
                       'comparator_calls': sum(d['comparator_calls'] for d in decisions),
                       'cost_scope': 'terminal acceptance; tool calls already incurred'}
