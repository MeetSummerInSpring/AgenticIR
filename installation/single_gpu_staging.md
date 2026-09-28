# Live single-GPU experimental deployment

`eval.run_manifest --staged-gpu GPU_UUID` attaches on-demand loading to the
existing local DepictQA routes. The severity and comparison services are mutually
exclusive and released before a restoration tool runs. Actual requests are sent
to the original services; no cached labels or precomputed plan are used.

The controller manages only services whose ownership records it created. A busy
GPU or occupied port causes rejection rather than termination of unknown work.
Model loading has a bounded startup timeout and at most 12 starts per sample.
Every start/stop and its elapsed time is saved in `local_services/switches.json`.

An optional `--catalog catalog.json` explicitly maps **every** registered task to
available tool names. It changes the declared deployment inventory, not the
perception or planning policy, and is recorded in configuration and episode
logs. For fair method comparisons, use the same catalog and budgets. A successful
small run establishes feasibility only; it is not a quality or speed comparison.

Example (the manifest and catalog are experiment files, not business data in Git):

```sh
python -m eval.run_manifest --manifest DEV_MANIFEST.csv --out NEW_OUTPUT \
  --method B1_current --mode auto --limit 2 --memory-read off \
  --max-tool-calls 6 --max-llm-calls 8 --timeout 1800 \
  --staged-gpu GPU_UUID --catalog CATALOG.json
```

Do not use `--plan` in auto mode. Keep production memories and frozen test inputs
out of this development run. External planning uses the configured authorized
text model only; local image paths never become external image payloads.

The optional `eval.perception_probe` is a separate development diagnostic. Supply
an ignored local JSON case list (`sample_id`, `domain`, `roi` in normalized xyxy,
and optional `path`) to `prepare --manifest ... --cases ... --out NEW_DIRECTORY`.
Run it only while the owned comparison service is ready. It compares a known
local blur against its source using full/ROI views and two prompts, including
same-image controls and swapped positions. Its labels are not expert quality
scores or a training/evaluation ground truth for real restoration.
