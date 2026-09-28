# UDR-S2Former local tool configurations

Author: https://github.com/Ephemeral182/UDR-S2Former_deraining (ICCV 2023).
Pinned commit and checkpoint SHA256 values are in
`executor/deraining/configs/udr_provenance.json`.

Clone the author repository into `executor/deraining/tools/UDR-S2Former` and check
out the recorded commit. The author repository contains its `pretrained` weights.
Do not substitute unrelated model-hub checkpoints. Vendor source and weights are
ignored by Git; the author repository does not provide a top-level license, so
this integration does not redistribute either.

The two tools `udr_raindrop_real` and `udr_agan` share one architecture. Their
weights respectively target the author's real joint-rain dataset and attached
raindrops. Both are appended to the deraining toolbox without priority overrides.
They are development candidates; installation is not evidence of task benefit.

The adapter follows the existing directory-based `Tool` contract and reuses the
`ridcp` environment (tested Python 3.8, torch 2.0.1+cu118, timm 1.0.16, einops).
Override its environment name with `AGENTICIR_UDR_ENV`. It runs strict state-dict
loading, RGB [0,1], evaluation mode, 320-pixel tiles and 64-pixel overlap, uniform
merging, and rounded/clamped 8-bit output. No resize or test-time augmentation.
Unlike the author's general test loader, inputs smaller than 320 are padded and
then unpadded; the development evaluation uses inputs larger than 320. The adapter
sets the author's module device explicitly; no vendor source is patched.

Use the bounded fixed-tool runner with a development manifest, for example:

```sh
python -m eval.tool_study --manifest DEV_MANIFEST.csv --out NEW_OUTPUT \
  --tools udr_raindrop_real udr_agan --limit 6 --split dev
```

Keep evaluation services stopped during tool execution on a single GPU. Full
execution identities and failures remain in the chosen output directory.
