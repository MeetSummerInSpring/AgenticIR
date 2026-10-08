import copy
from collections import defaultdict
from utils.episode_store import ToolStatistics


class FrozenStatistics:
    def __init__(self, store=None, context=None):
        self.context=context;self.scopes=set();self.scoped_statistics={}
        grouped=defaultdict(list)
        if store is not None and hasattr(store,'iter_events'):
            for event in store.iter_events(event_type='tool_attempt'):
                state=event['state'];action=event['action']
                if 'scale_long_edge' in state and 'observed_rain_severity' in state:
                    key=(action.get('subtask'),state['scale_long_edge'],state['observed_rain_severity'])
                    self.scopes.add(key)
                    if isinstance(action.get('tool'),str):grouped[(key,action['tool'])].append(event)
        for (key,name),events in grouped.items():
            outcomes=[e.get('outcome',{}) for e in events]
            durations=[e.get('transition',{}).get('duration_seconds') for e in events]
            durations=[v for v in durations if isinstance(v,(int,float)) and not isinstance(v,bool) and v>=0]
            stat=ToolStatistics(name,len(events),sum(e.get('status')=='execution_error' for e in outcomes),sum(isinstance(e.get('quality_success'),bool) for e in outcomes),sum(e.get('quality_success') is True for e in outcomes),sum(durations)/len(durations) if durations else None)
            self.scoped_statistics.setdefault(key,{})[name]=stat
        self.statistics={s:copy.deepcopy(store.get_tool_statistics(s)) for s in ['super-resolution','denoising','deraining','dehazing','brightening','motion deblurring','defocus deblurring','jpeg compression artifact removal']} if store else {}

    def get_tool_statistics(self, subtask):
        if self.context is not None and self.scopes:
            current=self.context();key=(subtask,current.get('scale_long_edge'),current.get('observed_rain_severity'))
            # A matching scope is insufficient: aggregate ONLY its own events.
            return copy.deepcopy(self.scoped_statistics.get(key,{}))
        return copy.deepcopy(self.statistics.get(subtask,{}))
