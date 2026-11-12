# Ascend dispatch optimization

Remote: <REMOTE_HOST>, container sglang-fby, /tmp/weight-loop4-validation.

Original remains test_loop4.py. Optimized implementation: test_loop4_tuned.py.
Run from parent of weight: python weight/validate_loop4_tuned.py.

Saved workload: N=3168, L=64, MC=72; autotune BLOCK=32 (candidates 16/32/64/128).
Unprofiled NPU-event complete call: baseline 128.06–128.78 us; tuned 50.72–51.01 us (~2.52x).
msprof original dispatch kernel: 9.888 us average. Combined comparison: original 9.892 us; fused kernel 41.722 us (includes tuning candidates).
Thus the full-call speedup comes from fusion and removing host launch overhead, not a faster isolated kernel. Original also launches two ReduceSum, two SelectV2, Cast, GreaterEqual and Greater per call.
Autotune adds one-time compilation/timing; exclude first call. Profile comparison includes tuning (ZerosLike traffic), so aggregate shares do not represent steady state.

Implementation removes explicit copy and persistent block loops, using a token/copy matrix, parallel reduction and associative prefix scan. One program per token tile. Preprocessing is fused into dispatch, but repeats row processing per token.

Validation: all 3168 saved outputs match baseline. Random N=0,1,31,129,3168 with zero-probability fallback and invalid maps match baseline. Parallel reduction/scan changes floating point accumulation order; exact CDF-boundary equivalence is not guaranteed. This is an experimental alternative, not a proven bitwise replacement for every input.

Profile directories in remote container: profile_baseline and profile_compare.
Reproduce: msprof --application="python weight/validate_loop4_tuned.py" --output=profile_compare_new

Next useful optimization: compute row CDF once, then use parallel comparisons or a logarithmic search for each token, avoiding repeated per-token prefix scans. Benchmark preprocessing separately and preserve boundary semantics before adopting.

## Updated timing contract: preprocessing excluded

Both dispatch functions now accept optional `prepared=(w,s)`. The comparison script prepares tables once before warmup/timing. test_loop4.py's main also passes prepared tables to every timed call. If omitted, dispatch still prepares tables for compatibility.

Measured with saved inputs: baseline times_us ~40.3–41.7 us; tuned ~56.8–57.8 us, BLOCK=64, vector_num=64. All 3168 outputs match. Under this corrected timing contract the current parallel scan version is slower than baseline; earlier 2.5x full-call speedup does not apply. The tuned kernel no longer computes row weights/sums; it still computes a per-token prefix scan.

## Sequential CDF + binary search revision

Latest test_loop4_tuned.py replaces per-token prefix scans with `prepare(prob,map)` outside timing. It builds CDF using the baseline's sequential fp32 accumulation, then binary-searches the CDF with 7 unrolled steps for MC=72. CDF/map/sum tables are loaded to UB. Grid is capped by vector_num. Autotune searches BLOCK=64/128/256/512, then caches the compiled Ascend launcher to skip repeated autotune/JIT dispatch in timed calls. The cached launcher is specific to this Triton-Ascend ABI.

Same-process measured times_us: baseline 39.37–39.64 us, tuned 26.48–27.15 us, selected BLOCK=128, approximately 1.48x. This improvement combines the algorithm and host launch changes; it is not an isolated kernel-speed claim. Earlier measurements above describe superseded versions.

Use `python weight/validate_loop4_tuned.py` from the parent directory. Pass `prepare(prob,mapping)` to the tuned dispatch; the baseline's `(weights,sums)` tuple is not interchangeable with the tuned `(cdf,sums)` tuple. Rebuild prepared tables when probabilities or mappings change. Binary search requires nonnegative weights and finite [0,1] random values, matching the probability-input contract. The table-in-UB implementation is validated for L=64, MC=72, not arbitrary large layouts.

## Current timing contract: all preparation included

Supersedes the exclusions above: benchmark calls dispatch without prepared tables. Both `_prepare_tables` and optimized CDF construction run on every timed call. CDF construction now uses one sequential-fp32 Triton kernel rather than 72 Python torch iterations.
Same-process saved-input results: baseline 127.97–133.49 us; tuned 147.45–151.80 us. Outputs match before and after the cached-launch benchmark. Thus the CDF binary-search version loses when preparation is included; the earlier 26–27 us result applies only with preparation excluded and must not be used as the current end-to-end speed claim.

## Baseline changed to test_loop3.py

validate_loop4_tuned.py now imports baseline and bench from unmodified test_loop3.py. Preparation remains included. Same-process saved-input measurement: loop3 46.11–46.63 us, CDF tuned 142.69–157.32 us. All 3168 outputs match on saved input, including cached launch. loop3 uses parallel cumsum with its last element as threshold total, whereas loop4/CDF uses torch row sum plus sequential prefixes; equality on supplied data does not establish identical floating-point boundary behavior for all inputs. Current CDF implementation does not beat the new baseline.

## Latest: test_loop3_tuned.py (loop3 baseline, all work timed)

Preserves loop3's fused probability/fallback/cumsum/threshold algorithm. Adds BLOCK autotune (8/16/32/64), uses vector_num-bounded grid, removes heuristic pick_block in the optimized copy, caches device metadata and the compiled Ascend launcher. No weight/CDF preparation is moved outside timing. test_loop3.py remains unmodified.

Saved workload: selected BLOCK=16; baseline mostly 45.29–45.94 us (one 47.98 us sample), tuned 43.11–43.22 us, approximately 5% improvement. This is mostly launch-path optimization, not a claimed algorithmic/kernel speedup. The compiled launcher follows the installed Triton-Ascend ABI. UB-resident strategy validated for supplied 64x72 layout; larger layouts require separate validation.

Reproduce: docker exec -w /tmp/weight-loop4-validation sglang-fby python weight/validate_loop3_tuned.py

## Joint UB tuning

Autotune now jointly searches BLOCK in {8,16,32,64} and PROB_UB/MAP_UB in {false,true}, 16 configurations. Both flags are retained in cached compiled-launch arguments. Latest saved-input winner: BLOCK=32, PROB_UB=true, MAP_UB=false. Same-process base times_us 45.38–45.93 us; tuned 35.17–35.34 us (about 22.6% lower). Saved outputs and cached-launch outputs match baseline. No computation moved outside timing.
