"""Summarize fixed VLM runs without any new model selection or label inference."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from .vlm_training import save_json

LABELS=('A','B','Tie','Uncertain')


def summarize(records):
    labeled=[r for r in records if r.get('answer') in LABELS]
    acc=lambda rows:sum(r['prediction']==r['answer'] for r in rows)/len(rows) if rows else None
    cls={c:acc([r for r in labeled if r['answer']==c]) for c in LABELS}
    present=[v for v in cls.values() if v is not None]
    groups=defaultdict(list);pairs=defaultdict(list);kinds=defaultdict(list)
    for r in records:
        groups[r.get('group_id','unknown')].append(r)
        pairs[r['pair_id']].append(r);kinds[r.get('kind','unknown')].append(r)
    norm=lambda r: (('left' if (r['prediction']=='A')==(r['order']=='forward') else 'right')
                    if r['prediction'] in ('A','B') else r['prediction'])
    paired=[rs for rs in pairs.values() if len(rs)==2 and {r['order'] for r in rs}=={'forward','reverse'}]
    consistent=sum(all(r['prediction']!='invalid' for r in rs) and len({norm(r) for r in rs})==1 for rs in paired)
    ga=[acc([r for r in rs if r.get('answer') in LABELS]) for rs in groups.values()];ga=[v for v in ga if v is not None]
    return dict(n_qa=len(records),n_labeled=len(labeled),n_pairs=len(pairs),n_primary_groups=len(groups),
        correct=sum(r['prediction']==r['answer'] for r in labeled),accuracy=acc(labeled),
        macro_accuracy=sum(present)/len(present) if present else None,
        group_equal_accuracy=sum(ga)/len(ga) if ga else None,
        binary_accuracy=acc([r for r in labeled if r['answer'] in ['A','B']]),
        class_accuracy=cls,kind_accuracy={k:acc([r for r in rs if r.get('answer') in LABELS]) for k,rs in kinds.items()},
        swap_consistency=consistent/len(paired) if paired else None,swap_consistent=consistent,swap_pairs=len(paired),
        invalid=sum(r['prediction']=='invalid' for r in records),predicted_counts=dict(Counter(r['prediction'] for r in records)))


def build(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train_dirs={'mixed':root/'train_aligned_mixed','controlled':root/'train_aligned_controlled','fresh':root/'train_new_sources'}
    rows=[];detailed={}
    def add(domain,arm,protocol,path):
        stat=summarize(json.loads(path.read_text()));key=domain+'/'+arm+'/'+protocol;detailed[key]=stat
        rows.append(dict(domain=domain,arm=arm,protocol=protocol,**{k:v for k,v in stat.items() if not isinstance(v,dict)},
            A_accuracy=stat['class_accuracy']['A'],B_accuracy=stat['class_accuracy']['B'],Tie_accuracy=stat['class_accuracy']['Tie'],Uncertain_accuracy=stat['class_accuracy']['Uncertain']))
    for arm,folder in train_dirs.items():
        domain='new_val' if arm=='fresh' else 'old_val'
        tag='epoch_'+str(json.loads((folder/'result.json').read_text())['best_epoch'])
        add(domain,arm,'free_four_choice',folder/(tag+'_predictions.json'))
        if arm in ['mixed','fresh']:add(domain,'baseline','free_four_choice',folder/'baseline_predictions.json')
    for arm in ['baseline','mixed','controlled','fresh']:
        folder=root/'postselection'/arm
        for domain in ['val','new_test','generic','tianlian']:
            for protocol,suffix in [('free_four_choice','_predictions.json'),('option_likelihood','_choice_predictions.json'),('native_prompt','_native_predictions.json')]:
                path=folder/(domain+suffix)
                if path.exists():add('old_val' if domain=='val' else domain,arm,protocol,path)
    with (out/'before_after.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(out/'metrics_detail.json',detailed)
    # Normality is an AI-assisted source tag, never a human preference label.
    source_rows=list(csv.DictReader((root/'new_final_test/data/sources.csv').open(encoding='utf-8-sig')))
    normal_ids={s['sample_id'] for s in source_rows if s['original_role']=='base'}
    normal_checks={}
    for arm in ['baseline','fresh']:
        for protocol,suffix in [('free_four_choice','_predictions.json'),('option_likelihood','_choice_predictions.json')]:
            records=json.loads((root/'postselection'/arm/('new_test'+suffix)).read_text())
            subset=[v for v in records if v['sample_id'] in normal_ids and v['answer']=='Tie']
            normal_checks[arm+'/'+protocol]=summarize(subset)
    save_json(out/'normality_diagnostic.json',dict(normal_source_ids=sorted(normal_ids),results=normal_checks,
        limitation='AI-assisted relative-normal tags; exact-identity tie only, not proof of protection against real tool overprocessing'))

    identities={name:dict(result=json.loads((p/'result.json').read_text()),history=json.loads((p/'history.json').read_text()),config=json.loads((p/'training_config.json').read_text())) for name,p in train_dirs.items()}
    save_json(out/'training_runs.json',identities)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(11,8));ax=axes.ravel()
    colors={'baseline':'#727b84','mixed':'#3875ad','controlled':'#8e62a8','fresh':'#2b8b69'}
    for name,p in train_dirs.items():
        steps=[json.loads(s) for s in (p/'training_steps.jsonl').read_text().splitlines()]
        ax[0].plot([s['optimizer_step'] for s in steps],[s['loss'] for s in steps],label=name,alpha=.65,color=colors[name])
    ax[0].set(xlabel='Optimizer updates',ylabel='Mean response token loss',title='Training (loss is not quality)');ax[0].legend()
    x=np.arange(3);width=.34
    for i,arm in enumerate(['baseline','fresh']):
        vals=[detailed[d+'/'+arm+'/option_likelihood']['class_accuracy']['Tie'] for d in ['new_test','generic','tianlian']]
        ax[1].bar(x+(i-.5)*width,vals,width,label=arm,color=colors[arm])
        vals=[detailed['new_test/'+arm+'/option_likelihood']['class_accuracy'][c] for c in LABELS]
        ax[2].bar(np.arange(4)+(i-.5)*width,vals,width,label=arm,color=colors[arm])
    ax[1].set(xticks=x,xticklabels=['New road (26)','HQ (24)','Tianlian (8)'],ylim=(0,1),ylabel='Exact-identity Tie accuracy',title='Tie retention: same option likelihood')
    ax[2].set(xticks=np.arange(4),xticklabels=LABELS,ylim=(0,1),ylabel='Class accuracy',title='New road: 78 / 78 / 26 / 26 QA');ax[2].legend()
    native=[detailed['new_test/'+arm+'/native_prompt']['accuracy'] for arm in ['baseline','fresh']]
    ax[3].bar(['baseline','fresh'],native,color=[colors['baseline'],colors['fresh']])
    ax[3].set(ylim=(0,1),ylabel='AB accuracy',title='Unchanged native prompt: 156 QA')
    for i,v in enumerate(native):ax[3].text(i,v+.015,f'{100*v:.2f}%',ha='center')
    fig.suptitle('Constructed-content diagnostics; option scores retain answer priors; no human MOS claim',fontsize=10)
    fig.tight_layout();fig.savefig(out/'training_validation.png',dpi=170);plt.close(fig)
    return detailed

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.root,a.out)
