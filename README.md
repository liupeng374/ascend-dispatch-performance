# Ascend Triton dispatch performance experiments

Standalone snapshots of a probability-based expert dispatch benchmark, optimized for the tested Triton-Ascend environment.

## Run

Requires an Ascend NPU environment with PyTorch NPU and Triton Ascend. Run from repository root:

```bash
python weight/validate_loop3_compact.py
```

This compares the unmodified `test_loop3.py` baseline, `test_loop3_tuned.py` (joint BLOCK/UB tuning), and `test_loop3_compact.py` (reduced tile padding). All data-dependent processing is timed; first-time compilation/autotuning is warmed up. Measurements use 20 warmups, 200 calls per sample, and 10 samples, in microseconds per call.

The bundled `.pt` files are the benchmark inputs. The compact implementation and cached compiled launch path are specific to the tested Triton-Ascend version; they are not a portable guarantee across devices or releases.

## Results and diagnosis

See [bottleneck report](weight/LOOP3_BOTTLENECK.md) for controlled ablations, loop experiments, UB limitations, and results. `PERFORMANCE.md` records historical experiments; later entries supersede earlier timing contracts.

Final representative same-process ranges: baseline 43.56–44.58 us; joint-tuned typically 35.23–35.36 us; compact 28.26–28.52 us. Results depend on hardware and runtime conditions.

Raw CSV and compiler IR evidence are in `weight/analysis_results/` and `weight/msprof_results/`. Historical loop4 experiments are retained for reference.

```bash
msprof --application="python weight/analyze_loop3.py" --output=profile_ablation
msprof --application="python weight/analyze_loop3.py --compact" --output=profile_ablation72
```

Code snapshots derive from the accompanying SGLang workspace; upstream licensing continues to apply. See LICENSE.
