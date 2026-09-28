import copy

class FrozenStatistics:
    def __init__(self, store=None, context=None):
        self.context = context
        self.scopes = set()
        if store is not None and hasattr(store, 'iter_events'):
            for event in store.iter_events(event_type='tool_attempt'):
                state = event['state']
                if 'scale_long_edge' in state and 'observed_rain_severity' in state:
                    self.scopes.add((event['action'].get('subtask'), state['scale_long_edge'], state['observed_rain_severity']))
        self.statistics = {s: copy.deepcopy(store.get_tool_statistics(s)) for s in
            ['super-resolution','denoising','deraining','dehazing','brightening','motion deblurring','defocus deblurring','jpeg compression artifact removal']} if store else {}

    def get_tool_statistics(self, subtask):
        if self.context is not None and self.scopes:
            current = self.context()
            key = (subtask, current.get('scale_long_edge'), current.get('observed_rain_severity'))
            if key not in self.scopes:
                return {}
        return copy.deepcopy(self.statistics.get(subtask, {}))
