# Expanded data, terminal acceptance and local weather context

These switches are experimental. Production defaults and model weights are unchanged.

## Data admission

Keep existing manifest fields. Add `source_id`, `source_batch`, `parent_id`,
`asset_type`, `camera_id`, `weather_context_id` and `prior_use`. Normalize aliases
and weather episodes before splitting; an absent alias is not proof of independence.
Use `reference_sample_id` and/or `reference_sha256` for reference lineage.

```bash
python -m eval.audit_expansion --manifest increment.csv \
  --prior prior_exposure.json --out admission.json
```

The audit rejects cross-split source, group, parent, reference and known pixel
components. New final-test rows need `prior_use=confirmed_project_unseen` and no
historical component overlap. Camera overlap is separately reported, not mislabeled
as camera-held-out generalization. Count original observations, derived images,
source groups, cameras, preference pairs and QA exposures separately.

## Terminal acceptance

Add `--acceptance prefix_bidirectional` to `python -m eval.run_manifest`.
Default `original` reproduces the previous acceptance path. The experimental gate
runs **after** search/rollback, traversing the selected trajectory from input through
intermediate images to final output. A challenger replaces the incumbent only when
both native comparison orders agree. Exact pixel duplicates need no comparison.
Order disagreement retains the incumbent; it is not a trained uncertainty label.

`agent_summary.json` records the original final image, accepted image, every pair,
both answers and request count. The selected prefix is reflected in the saved
execution path. All actual tool invocations remain charged, including discarded
results. Fixed-candidate replay is a separate experiment, never a tool-cost saving.
The gate can retain useful intermediate restoration, but can also reject beneficial
processing. Evaluate original gate / new gate with original comparator / new gate
with trained comparator on the same candidate pixels before wider testing.

## Weather stays local

`--weather-contexts contexts.json --weather-mode offline_association` selects an
optional JSON object keyed by the manifest's `weather_context_id`. Each raw context
contains station ID, mapping basis, timezone-aware image/record time, optional
observation interval and `available_at`, `elements` with name/value/unit, and missing
fields. Unknown values stay unknown. Raw observations and their summaries are **not
sent to the external text planner**. The base visual prompt remains unchanged;
`local_schedule_advice` receives validated observations locally and logs its action.

Validated positive precipitation may move an already visually proposed `deraining`
task first. It never adds a task or converts station rain into an image label.
Compass wind alone has no supported ordering change. This is a minimal deterministic
local advisory policy, not a trained meteorological planner or demonstrated benefit.

Missing or uninterpretable context returns the pure-vision plan. Synthetic inputs
withhold source weather. Real-time mode requires confirmed availability at image
time; precipitation also needs known interval bounds. Future observations or unknown
publication lag are not real-time eligible. An unknown interval is `null`, not an
invented instantaneous measurement.

`eval.import_weather_contexts` retains a source hash and raw context reference. For
the supplied tables with unspecified numeric units, it extracts only explicit
16-sector compass direction from `2分钟平均风_风向`. `compass16` names a categorical
representation, not an inferred physical unit. Rain, temperature, humidity, pressure,
speed and visibility remain unused until their units are confirmed. Record weather
association, local-advisor consumption, final plan and costs separately. A no-change
result is valid and does not establish weather-context effectiveness.

## Restoration preference pilot

`eval.prepare_restoration_preferences` compares actual restoration candidates using
the pre-perturbation source observation as a reference. PSNR/SSIM/LPIPS must all agree
beyond frozen margins. These are explicitly **objective pseudo labels**, not human
quality truth; underlying road images can already contain rain or other damage.
Metric disagreements/near ties stay pending review, not supervised `Uncertain`.
Sparse exact-identity and metadata-proven noncorrespondence controls preserve Tie
and Uncertain prompt behavior. Reuse `eval.vlm_training`; select epochs on validation
only and compare original / prior-trained / expanded-trained with identical native
and four-choice prompts. A pilot with old source groups does not fulfill a planned
new-source expansion and must be reported separately.
