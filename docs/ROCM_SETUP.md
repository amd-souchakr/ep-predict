# Single-MI355X ROCm environment

## Scope

This environment preserves the project's existing execution path:

- Hugging Face Transformers loads the unmodified model;
- project hooks observe the explicit OLMoE routers;
- PyTorch executes BF16 inference on one MI355X;
- H4 uses PyTorch events and pinned asynchronous copies through the same
  `torch.cuda` API used by ROCm.

It does not add vLLM or fused serving kernels because those can bypass the
module hooks whose semantics are under study. It also does not introduce
tensor parallelism, expert parallelism, or other multi-GPU behavior. Those
choices keep new results comparable with the earlier single-NVIDIA-GPU runs
while changing the hardware backend.

## Frozen software target

| Component | Project target | Reason |
|---|---:|---|
| Python | 3.12 | Existing project contract |
| uv | 0.5.3 or newer | Required by uv's PyTorch source configuration |
| Host ROCm | 7.2 | Installed MI355X software stack |
| PyTorch | 2.11.0+rocm7.2 | Official ROCm 7.2 wheel for Python 3.12 |
| Transformers | 5.14.1 | Existing pinned model implementation |
| Model dtype | BF16 | Native on `gfx950` and consistent with prior runs |

`pyproject.toml` marks the ROCm index explicit. Only `torch`,
`pytorch-triton-rocm`, and `triton-rocm` may resolve from it; ordinary Python
packages remain on PyPI. On Linux this prevents a CUDA PyTorch wheel and its
`nvidia-*` dependencies from entering the lock or environment.

## Host boundary

The host administrator or machine image must already provide a compatible AMD
driver, ROCm user space, and access to `/dev/kfd` and `/dev/dri`. This project
does not use `sudo` and does not install native OS packages.

These read-only checks establish the boundary before Python setup:

```bash
cat /opt/rocm/.info/version
rocm-smi --showdriverversion --showproductname
rocminfo | grep -m 1 gfx950
test -r /dev/kfd && test -w /dev/kfd
test -d /dev/dri
```

If the device-node checks fail, stop there and run the experiment from a job,
container, or shell that has GPU device access. A Python package cannot repair
missing device nodes or host permissions.

## Create the environment

From the repository root:

```bash
uv sync --extra inference
```

The resolved lock should identify ROCm and contain no CUDA runtime packages:

```bash
uv export --extra inference --frozen --no-hashes \
  | grep -E '^(torch|pytorch-triton-rocm|triton-rocm|nvidia-)'
```

## Qualify one GPU without downloading model weights

Always constrain visibility for this experiment family:

```bash
HIP_VISIBLE_DEVICES=0 uv run ep-predict verify-rocm
```

The command fails unless all of these hold:

1. the installed PyTorch wheel reports ROCm and no CUDA runtime;
2. exactly one accelerator is visible;
3. the visible architecture is `gfx950`;
4. a BF16 matrix multiplication completes with finite output;
5. a pinned asynchronous host-to-device copy preserves its data;
6. a tiny random OLMoE forward produces six valid routing records with no
   hook/top-k mismatch.

The output is JSON so it can be retained with machine qualification records.

## Project tests

Run the complete dependency set and unit/integration suite:

```bash
uv sync --all-extras
HIP_VISIBLE_DEVICES=0 uv run python -m unittest discover -s tests -v
```

The existing hook integration test uses a tiny random OLMoE on CPU. The ROCm
verifier is the corresponding GPU execution test.

## Pinned-checkpoint end-to-end smoke

The first command downloads and loads the pinned 13.8 GB BF16 checkpoint. The
second exercises tokenization, generation, router hooks, trace validation, and
request-level artifact writing for one prompt.

```bash
HIP_VISIBLE_DEVICES=0 uv run ep-predict inspect \
  --config configs/model/olmoe-1b-7b-instruct.toml \
  --output artifacts/mi355x-model-report.json

uv sync --extra data --extra inference
uv run ep-predict prepare-dataset \
  --config configs/dataset/h1-standard-small.toml

HIP_VISIBLE_DEVICES=0 uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/h1-pilot.toml \
  --limit 1
```

Do not compare MI355X timing against the old NVIDIA calibration until H4 is
remeasured on the AMD host. Trace-derived H1-H3 metrics should reproduce for
the same model, tokens, and deterministic generation, while H4 transfer and
decode timing are explicitly hardware-specific.

## Experiment controls

- Keep `HIP_VISIBLE_DEVICES=0` in every inference and predictor command.
- Leave config devices as `cuda:0`; `hip:0` is not a valid PyTorch device.
- Record `environment_report()` with every run. It now includes the PyTorch
  backend, HIP version, GPU architecture, and visibility variables.
- Warm up kernels before timing. Never include the first compilation-bearing
  forward in H4 measurements.
- Use SDPA for the initial qualification. Add an optimized attention or fused
  MoE path only as a separately qualified model-specific change, with proof
  that the router hooks still observe the dispatched expert IDs.
