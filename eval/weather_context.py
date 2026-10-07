"""Validate optional observations and expose only anonymous numeric planner data.

Raw context stays in the local run record. Observation values never become image
degradation labels. The interface only helps order an existing visual agenda.
"""
import datetime as dt
import json
import math

UNITS = {'precipitation': {'mm'}, 'temperature': {'degC'},
         'relative_humidity': {'%'}, 'visibility': {'m', 'km'},
         'wind_speed': {'m/s'}, 'wind_direction': {'deg', 'compass16'}}
COMPASS16 = {'N','NNE','NE','ENE','E','ESE','SE','SSE',
             'S','SSW','SW','WSW','W','WNW','NW','NNW'}


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone required')
    return parsed


def prepare_context(raw, role='real', mode='realtime'):
    if mode not in {'realtime', 'offline_association'}:
        raise ValueError('unknown weather mode')
    result = {'usable': False, 'mode': mode, 'reason': 'missing', 'planner_context': None}
    if not raw:
        return result
    if role != 'real':
        return dict(result, reason='non_real_input_weather_withheld')
    try:
        if not raw.get('station_id') or not raw.get('mapping_basis'):
            raise ValueError('station/mapping basis missing')
        image = timestamp(raw['image_time'])
        record = timestamp(raw['record_time'])
        end = timestamp(raw['interval_end']) if raw.get('interval_end') else None
        start = timestamp(raw['interval_start']) if raw.get('interval_start') else None
        if start is not None and end is not None and start > end:
            raise ValueError('reversed observation interval')
        available = timestamp(raw['available_at']) if raw.get('available_at') else None
        if available is not None and available < record:
            raise ValueError('availability before observation')
        if mode == 'realtime' and (available is None or max(record, end or record, available) > image):
            return dict(result, reason='not_known_available_at_image_time')
        elements = []
        rejected = 0
        for e in raw.get('elements', []):
            name, unit, value = e.get('name'), e.get('unit'), e.get('value')
            if name == 'wind_direction' and unit == 'compass16' and isinstance(value, str) and value in COMPASS16:
                elements.append({'name':name, 'unit':unit, 'value':value})
                continue
            if (name not in UNITS or unit not in UNITS[name] or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)):
                rejected += 1
                continue
            if name == 'precipitation' and mode == 'realtime' and (start is None or end is None):
                rejected += 1
                continue
            elements.append({'name': name, 'unit': unit, 'value': value})
        if not elements:
            return dict(result, reason='no_interpretable_elements')
        # No station/camera/sample identities, exact timestamps, paths or raw text
        # are inserted into the external text planner.
        anonymous = {'mode': mode, 'elements': elements,
                     'record_age_seconds': (image-record).total_seconds(),
                     'interval_end_age_seconds': (image-end).total_seconds() if end else None,
                     'interval_duration_seconds': (end-start).total_seconds() if end and start else None,
                     'availability_known': available is not None,
                     'missing_fields_count': len(raw.get('missing_fields', [])),
                     'rejected_elements_count': rejected}
        return dict(result, usable=True, reason='validated_observation', planner_context=anonymous)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        # Do not export malformed raw values or exception contents into prompts.
        return dict(result, reason='invalid_context_' + type(exc).__name__)


def planner_postscript(context):
    if not context['usable']:
        return ''
    return ('\nOptional station observations (not image degradation labels): ' +
            json.dumps(context['planner_context'], sort_keys=True, allow_nan=False) +
            '\nUse only to order the existing visual agenda. Do not add a task or '
            'infer instantaneous image rain from interval precipitation. '
            'Offline association does not establish real-time availability.\n')


def local_schedule_advice(plan, context):
    """Local-only optional ordering; never export observations to a remote model.

    Positive precipitation may prioritize an ALREADY visually proposed derain
    task. No numeric intensity threshold or added degradation label is inferred.
    Compass-only context has no defensible ordering rule without camera geometry.
    """
    if not context['usable']:
        return list(plan), 'visual_fallback_' + context['reason']
    precipitation=[e for e in context['planner_context']['elements']
                   if e['name']=='precipitation' and e['unit']=='mm']
    if len(plan)>1 and 'deraining' in plan and any(e['value']>0 for e in precipitation):
        return ['deraining']+[t for t in plan if t!='deraining'], 'observed_precipitation_prioritizes_existing_visual_task'
    return list(plan), 'no_supported_ordering_change'
