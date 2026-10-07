import copy
import unittest
from .candidate_acceptance import accept_prefix, trajectory_prefix
from .weather_context import prepare_context, planner_postscript


class AcceptanceTests(unittest.TestCase):
    def nodes(self):
        return [{'img_path': p} for p in ['input', 'middle', 'final']]

    def test_improvement_then_overprocessing(self):
        answers = iter(['latter', 'former', 'former', 'latter'])
        selected, record = accept_prefix(self.nodes(), lambda *args: next(answers), str)
        self.assertEqual(selected['img_path'], 'middle')
        self.assertEqual(record['comparator_calls'], 4)

    def test_disagreement_retains_then_can_accept(self):
        answers = iter(['latter', 'latter', 'latter', 'former'])
        selected, record = accept_prefix(self.nodes(), lambda *args: next(answers), str)
        self.assertEqual(selected['img_path'], 'final')
        self.assertEqual(record['decisions'][0]['outcome'], 'order_disagreement')

    def test_duplicate_and_noop_need_no_request(self):
        def fail(*args):
            self.fail('duplicate called comparator')
        selected, record = accept_prefix(self.nodes(), fail, lambda p: 'same_pixels')
        self.assertEqual(selected['img_path'], 'input')
        self.assertEqual(record['comparator_calls'], 0)

    def test_prefix_excludes_abandoned_branches(self):
        tree = {'img_path': 'input', 'children': {'rain': {'tools': {
            'failed': {'img_path': 'abandoned'}, 'chosen': {'img_path': 'final'}}}}}
        self.assertEqual([n['img_path'] for n in trajectory_prefix(tree, 'final')], ['input', 'final'])
        self.assertEqual(trajectory_prefix(tree, 'missing'), [])
        with self.assertRaises(ValueError):
            accept_prefix([], None)


class WeatherTests(unittest.TestCase):
    def fixture(self):
        return dict(station_id='PRIVATE_STATION', mapping_basis='PRIVATE_MAPPING',
                    image_time='2026-01-01T12:00:00+08:00',
                    record_time='2026-01-01T11:00:00+08:00',
                    interval_start='2026-01-01T10:00:00+08:00',
                    interval_end='2026-01-01T11:00:00+08:00',
                    available_at='2026-01-01T11:05:00+08:00',
                    elements=[dict(name='precipitation', value=2.5, unit='mm')])

    def test_whitelist_and_privacy(self):
        raw = self.fixture(); original = copy.deepcopy(raw)
        c = prepare_context(raw); prompt = planner_postscript(c)
        self.assertTrue(c['usable'])
        self.assertEqual(c['planner_context']['record_age_seconds'], 3600)
        self.assertNotIn('PRIVATE', prompt)
        self.assertNotIn('2026', prompt)
        self.assertEqual(raw, original)

    def test_future_and_unknown_availability(self):
        for key, value in [('interval_end','2026-01-01T13:00:00+08:00'),
                           ('available_at',None)]:
            raw = self.fixture(); raw[key] = value
            self.assertFalse(prepare_context(raw)['usable'])
            self.assertTrue(prepare_context(raw, mode='offline_association')['usable'])

    def test_synthetic_and_missing_fall_back(self):
        self.assertFalse(prepare_context(self.fixture(), role='synthetic')['usable'])
        self.assertEqual(planner_postscript(prepare_context(None)), '')

    def test_unknown_accumulation_window_is_not_instantaneous(self):
        raw=self.fixture();raw.pop('interval_start');raw.pop('interval_end')
        self.assertFalse(prepare_context(raw)['usable'])
        c=prepare_context(raw, mode='offline_association')
        self.assertTrue(c['usable'])
        self.assertIsNone(c['planner_context']['interval_duration_seconds'])

    def test_categorical_wind_keeps_units_unknown_for_numeric_fields(self):
        from .import_weather_contexts import normalize
        raw=self.fixture();raw['available_at']=None
        raw['elements']=[dict(name='2分钟平均风_风向',value='西西南',unit=None),
                         dict(name='气温',value='22.8',unit=None)]
        normalized,report=normalize({'example':raw})
        c=prepare_context(normalized['example'],mode='offline_association')
        self.assertEqual(c['planner_context']['elements'],[dict(name='wind_direction',unit='compass16',value='WSW')])
        self.assertFalse(prepare_context(normalized['example'])['usable'])

    def test_ambiguous_units_times_and_nonfinite_rejected(self):
        for value in [dict(name='precipitation',value=1,unit='unknown'),
                      dict(name='precipitation',value=float('nan'),unit='mm'),
                      dict(name='precipitation',value=True,unit='mm')]:
            raw=self.fixture();raw['elements']=[value]
            self.assertFalse(prepare_context(raw)['usable'])
        raw=self.fixture();raw['image_time']='2026-01-01T12:00:00'
        self.assertFalse(prepare_context(raw)['usable'])


class PlannerIntegrationTests(unittest.TestCase):
    def agent(self):
        from unittest.mock import Mock
        from .experiment_agent import ExperimentAgent
        agent=ExperimentAgent.__new__(ExperimentAgent)
        agent.weather_raw=WeatherTests().fixture()
        agent.weather_prepared=prepare_context(agent.weather_raw)
        agent.work_mem={};agent.with_retrieval=True
        agent.subtask_degra_dict={'deraining':'rain','brightening':'dark'}
        agent.schedule_memory=Mock()
        agent.schedule_memory.render_for_prompt.return_value=('frozen rules',[])
        agent.gpt4=Mock(return_value="{'thought':'fixture response','order':['deraining','brightening']}")
        agent.workflow_logger=Mock();agent._episode_state=Mock(return_value={})
        agent._append_episode_event=Mock()
        return agent

    def test_context_reaches_local_schedule_but_never_external_prompt(self):
        agent=self.agent()
        self.assertEqual(agent.schedule(['deraining','brightening']),['deraining','brightening'])
        prompt=agent.gpt4.call_args.kwargs['prompt']
        self.assertNotIn('precipitation',prompt)
        self.assertNotIn('PRIVATE',prompt)
        self.assertTrue(agent.work_mem['weather_decisions'][0]['consumed_by_local_advisor'])
        self.assertFalse(agent.work_mem['weather_decisions'][0]['sent_to_external_model'])

    def test_single_visual_task_does_not_force_weather_task_or_request(self):
        agent=self.agent()
        self.assertEqual(agent.schedule(['brightening']),['brightening'])
        agent.gpt4.assert_not_called()
        self.assertEqual(agent.work_mem['weather_decisions'][0]['advice_reason'],'no_supported_ordering_change')

    def test_plan_audit_survives_execution_and_agenda_mutation(self):
        agent=self.agent()
        agenda=['brightening']
        plan=agent.schedule(agenda)
        plan.pop(0)
        agenda.clear()
        record=agent.work_mem['weather_decisions'][0]
        for field in ['agenda','visual_plan','plan']:
            self.assertEqual(record[field],['brightening'])
        event=agent._append_episode_event.call_args.kwargs['outcome']
        self.assertEqual(event['plan'],['brightening'])
        self.assertEqual(plan,[])

    def test_local_prior_never_adds_a_task(self):
        from .weather_context import local_schedule_advice
        context=prepare_context(WeatherTests().fixture())
        plan,reason=local_schedule_advice(['brightening','deraining'],context)
        self.assertEqual(plan,['deraining','brightening'])
        self.assertEqual(local_schedule_advice(['brightening','denoising'],context)[0],['brightening','denoising'])


if __name__ == '__main__':
    unittest.main()
