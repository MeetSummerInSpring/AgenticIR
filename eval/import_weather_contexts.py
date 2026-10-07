"""Normalize explicitly interpreted fields while retaining an immutable raw input.

Only categorical compass wind direction can currently be interpreted without
physical units from the supplied Chinese table headers. Do not guess units for
rain, temperature, humidity, pressure, speed or visibility from plausible values.
"""
import argparse
import json
from pathlib import Path
from .weather_context import prepare_context
from .experiment_support import atomic_json
from .manifest_io import sha256

COMPASS = dict(zip(['北','北东北','东北','东东北','东','东东南','东南','南东南',
                   '南','南西南','西南','西西南','西','西西北','西北','北西北'],
                  ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']))


def normalize(raw):
    normalized={};report=[]
    for key,context in raw.items():
        c=dict(context);elements=[]
        # Prefer the explicitly named 2-minute mean. Do not merge observation
        # windows or turn a compass sector into a more precise numeric bearing.
        e=next((e for e in c.get('elements',[]) if e.get('name')=='2分钟平均风_风向'),None)
        if e and e.get('value') in COMPASS:
            elements.append({'name':'wind_direction','value':COMPASS[e['value']],
                             'unit':'compass16','raw_value':e['value'],
                             'aggregation':'2-minute mean; interval alignment unknown',
                             'unit_basis':'categorical compass vocabulary, not inferred physical unit'})
        c.update(elements=elements,raw_context_id=key,
                 normalization_note='Only unambiguous categorical wind direction; numeric units unconfirmed; offline association only')
        normalized[key]=c
        checked=prepare_context(c,mode='offline_association')
        report.append({'context_id':key,'usable':checked['usable'],'reason':checked['reason'],
                       'usable_fields':['wind_direction'] if elements else [],
                       'numeric_weather_fields_used':False})
    return normalized,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    raw=json.loads(Path(a.raw).read_text());normalized,report=normalize(raw)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'weather_contexts.json',normalized)
    atomic_json(out/'audit.json',{'raw_sha256':sha256(a.raw),'raw_path':str(Path(a.raw).resolve()),
        'raw_count':len(raw),'usable_offline_compass_contexts':sum(r['usable'] for r in report),
        'numerical_fields_with_confirmed_units':0,'real_time_eligible':0,'contexts':report})


if __name__=='__main__':main()
