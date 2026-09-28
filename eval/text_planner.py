"""Text-only DashScope transport, finite retries, and explicit usage accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import requests
import yaml

MODEL = 'deepseek-v4-flash-0731'
ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'


class ModelAccessError(RuntimeError):
    pass


def validate_text_payload(payload):
    if payload.get('model') != MODEL:
        raise ValueError('experiment model must be '+MODEL)
    for message in payload['messages']:
        content=message['content']
        blocks=[{'type':'text','text':content}] if isinstance(content,str) else content
        for block in blocks:
            if block.get('type')!='text': raise ValueError('external planner accepts text only')
            text=block.get('text','')
            if re.search(r'data:image|/root/|/home/|[A-Z]:\\|T_I[0-9a-f]{10}|S_V[0-9a-f]{10}|WLPD-|camera_full_code',text):
                raise ValueError('private path, image payload or sample identity in planner text')


class TextTransport:
    def __init__(self, log_path, max_calls=20, endpoint=ENDPOINT):
        self.log_path=Path(log_path);self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.block_path=self.log_path.with_suffix('.blocked.json')
        self.max_calls=max_calls;self.calls=0;self.endpoint=endpoint
        if endpoint != ENDPOINT: raise ValueError('use the existing regional endpoint; no silent endpoint substitution')

    def __call__(self, headers, payload):
        validate_text_payload(payload)
        if self.block_path.exists(): raise ModelAccessError('model access blocked; inspect '+str(self.block_path))
        for attempt in range(3):
            if self.calls>=self.max_calls: raise RuntimeError('text request budget exhausted')
            self.calls+=1;start=time.monotonic();response=None
            rec={'model':payload['model'],'endpoint':self.endpoint,'attempt':attempt+1,'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
            try:
                response=requests.post(self.endpoint,headers=headers,json=payload,timeout=(10,90))
                try: body=response.json()
                except ValueError: body={}
                rec.update(http_status=response.status_code,usage=body.get('usage',{}),request_id=body.get('id'),error=body.get('error'))
                if response.status_code in (401,403) or (response.status_code==400 and any(v in json.dumps(body).lower() for v in ['quota','balance','permission'])):
                    self.block_path.write_text(json.dumps(rec,ensure_ascii=False,indent=2))
                    raise ModelAccessError('model authorization/quota failure: '+str(response.status_code)+' '+str(body.get('error',{})))
                if response.status_code==429 and attempt<2:
                    time.sleep(2**attempt);continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                rec['transport_error']=type(exc).__name__
                raise
            finally:
                rec['duration_seconds']=time.monotonic()-start
                with self.log_path.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        raise RuntimeError('rate limit retries exhausted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yml');p.add_argument('--out',required=True);a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if cfg['GPT']['MODEL']!=MODEL: raise ValueError('local config model differs; create explicit experiment config first')
    payload={'model':MODEL,'messages':[{'role':'user','content':'Return only valid JSON: {"ready":true,"mode":"text_only"}. Do not include markdown.'}],'max_tokens':256,'temperature':0}
    transport=TextTransport(out/'usage.jsonl',max_calls=3)
    try:
        response=transport({'Authorization':'Bearer '+cfg['GPT']['API_KEY'],'Content-Type':'application/json'},payload)
        body=response.json();answer=body['choices'][0]['message']['content'];parsed=json.loads(answer)
        (out/'probe.json').write_text(json.dumps({'status':'ok','model':body.get('model'), 'requested_model':MODEL,'usage':body.get('usage'),'response':parsed},indent=2))
        print('text probe ok; usage',body.get('usage'))
    except Exception as exc:
        (out/'probe.json').write_text(json.dumps({'status':'failed','model':MODEL,'error':str(exc)},indent=2));print(type(exc).__name__,str(exc));raise SystemExit(1)
if __name__=='__main__': main()
