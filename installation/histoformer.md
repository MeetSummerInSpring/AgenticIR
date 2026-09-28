# Histoformer real-image restoration tool

`histoformer_real` is appended to the existing deraining toolbox and uses the
same `Tool` interface: one image in, one `output.png` out, subprocess timeout,
validated decoding, and per-call logs. Existing tool order is preserved. The
new tool has no calibrated historical priority. Recalibrate or version tool
profiles before making new architecture comparisons.

Author: Shangquan Sun et al., *Restoring Images in Adverse Weather Conditions
via Histogram Transformer*, ECCV 2024.

- Author code: https://github.com/sunshangquan/Histoformer
- Pinned code commit: `1f045f06c03551c31504d8042dbe6ff9b9569108`
- Author weight URL: https://huggingface.co/sunsean/Histoformer/resolve/f6acde20acaa26ef5e235f2c71ff7637f4d9a945/Allweather/pretrained_models/net_g_real.pth
- Weight SHA256: `8ae731d0fd5e2761cec7428322c7602146cba42c73188cc4b0fcc17232065958`
- Weight size: 66,603,665 bytes.

Clone author code under `executor/deraining/tools/Histoformer` and check out the
pinned revision; place the verified weight at
`Allweather/pretrained_models/net_g_real.pth` inside it. No vendor code or weights
are committed to this repository. The author's main Hugging Face model card
declares MIT; the GitHub checkout has no root LICENSE. This is a research
integration, and no independent blanket rights claim is made for training data.

The wrapper reuses the existing `ridcp` environment by default. Set
`AGENTICIR_HISTOFORMER_ENV` to another compatible Conda environment if needed.
Verified versions: Python 3.8, torch 2.0.1+cu118, einops 0.8.1, NumPy 1.24.4,
Pillow 10.4.0, PyYAML 6.0.2. Do not upgrade the main agent environment just to
install this tool. The adapter imports only the author architecture, not the
BasicSR training package. Inference performs no downloads.

Inference follows the author RGB float32 [0,1] convention and network config,
loads `params` strictly with `weights_only=True`, reflects to multiples of 8,
and unpads to original dimensions. No resizing, patch inference, ensembling,
retraining, or diffusion sampling is used. A replicate-padding fallback exists
only for dimensions below 8 pixels; the project uses much larger inputs.

The real-image checkpoint was chosen before project scores were inspected.
Its name does not guarantee domain transfer to large lens occlusion, glare,
nighttime ruler measurement, or physical readout accuracy. Keep the original
image as a candidate. `eval.real_acceptance` is a separate offline diagnostic;
it does not change production memory or silently enable an online safety claim.

For staged local DepictQA evaluation, retain the two loopback binding changes
in `installation/depictqa_localhost.patch` (use `git apply --unidiff-zero` once from the project root on
an unmodified matching vendor checkout). `eval.local_service` refuses to start
an unpatched public-listening entrypoint. Neither service is a cloud evaluator.
