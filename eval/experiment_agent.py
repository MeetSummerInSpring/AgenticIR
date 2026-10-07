"""Small isolated experiment adapter; production IRAgent defaults stay intact."""
import copy
from pathlib import Path
from pipeline.iragent import IRAgent
from utils.episode_store import EpisodeStore


from .frozen_statistics import FrozenStatistics
from .text_planner import TextTransport, MODEL


class RegistrySelector:
    def select(self, subtask, tools):
        return list(tools), [{'tool_name':t.tool_name,'selected':True,'policy':'registry_order'} for t in tools]


class ExperimentAgent(IRAgent):
    def __init__(self, *args, selector_policy='current', memory_read='frozen',
                 memory_snapshot=None, memory_update=False, log_events=True, max_tool_calls=8, max_llm_calls=20,
                 acceptance='original', weather_context=None, weather_role='real', weather_mode='realtime', **kwargs):
        from .weather_context import prepare_context
        if acceptance not in {'original', 'prefix_bidirectional'}:
            raise ValueError('unknown acceptance policy')
        self.acceptance_policy = acceptance
        self.weather_raw = weather_context
        self.weather_prepared = prepare_context(weather_context, weather_role, weather_mode)
        if kwargs.get('evaluate_degradation_by','depictqa') != 'depictqa' or kwargs.get('reflect_by','depictqa') != 'depictqa':
            raise ValueError('experiment vision must use local DepictQA; external transport is text-only')
        self.log_events = log_events; self.llm_calls=0; self.tool_calls=0; self.local_evaluation_calls=0
        super().__init__(*args, **kwargs)
        self.executor = copy.copy(self.executor)
        owner = self
        class BudgetedTool:
            def __init__(self, tool): self.tool=tool; self.tool_name=tool.tool_name
            def __call__(self, *args, **kwargs):
                if owner.tool_calls >= max_tool_calls: raise RuntimeError('tool call budget exhausted')
                owner.tool_calls += 1
                return self.tool(*args, **kwargs)
        self.executor.toolbox_router = {s:[BudgetedTool(t) for t in ts] for s,ts in self.executor.toolbox_router.items()}
        if self.gpt4.model != MODEL:
            raise ValueError('planner model differs from frozen experiment model')
        transport=TextTransport(self.work_dir/'text_usage.jsonl',max_calls=max_llm_calls)
        def send_request(headers,payload):
            try:return transport(headers,payload)
            finally:self.llm_calls=transport.calls
        self.gpt4._send_request=send_request
        if self.depictqa:
            local_post=self.depictqa.session.post
            def counted_post(*args,**kwargs):
                self.local_evaluation_calls+=1
                return local_post(*args,**kwargs)
            self.depictqa.session.post=counted_post
        if selector_policy == 'registry': self.tool_selector = RegistrySelector()
        elif not memory_update:
            source = EpisodeStore(Path(memory_snapshot)) if memory_read=='frozen' and memory_snapshot else None
            def current_context():
                from PIL import Image
                with Image.open(self.cur_node['img_path']) as im:
                    scale=max(im.size)
                return {'scale_long_edge':scale,'observed_rain_severity':self.cur_node.get('state',{}).get('rain')}
            self.tool_selector.episode_store = FrozenStatistics(source, context=current_context)
        elif memory_read == 'off':
            raise ValueError('online update requires memory reads')

    def _append_episode_event(self, **kwargs):
        if self.log_events: return super()._append_episode_event(**kwargs)
        return ''

    def schedule(self, agenda, ps=''):
        from .weather_context import local_schedule_advice
        context = self.weather_prepared
        # Weather observations, even anonymous summaries, remain local under the
        # data-transfer authorization. External prompts are identical to M0.
        visual_plan = super().schedule(agenda, ps)
        plan, reason = local_schedule_advice(visual_plan, context)
        # Execution consumes the returned plan; audit records need independent snapshots.
        record = dict(context, consumed_by_local_advisor=context['usable'],
                      sent_to_external_model=False, agenda=list(agenda),
                      visual_plan=list(visual_plan), plan=list(plan), advice_reason=reason)
        self.work_mem.setdefault('weather_decisions', []).append(record)
        self._append_episode_event(event_type='weather_context_decision',
                                   action={'context': self.weather_raw}, outcome=record)
        return plan

    def _record_res(self):
        if self.acceptance_policy == 'prefix_bidirectional':
            from .candidate_acceptance import trajectory_prefix, accept_prefix
            nodes = trajectory_prefix(self.work_mem['tree'], self.cur_node['img_path'])
            selected, record = accept_prefix(nodes, self.compare_quality)
            self.work_mem['acceptance'] = record
            self._append_episode_event(event_type='terminal_acceptance', outcome=record)
            self.cur_node = selected
        return super()._record_res()
