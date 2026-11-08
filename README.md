# bus-fill

Filling the memory bus on Volta — a batched Q4-K GEMM for sm_70, where the
Ampere-only kernels (Marlin et al.) refuse to run.

## Why

On a single modern card, a quantized-weight decode kernel reads the whole
weight set from memory once per step and the tensor cores sit ~96% idle at
batch size 1. Batching multiplies the math done per memory trip — which is
why vLLM on one RTX 3090 goes 45 → 298 tok/s combined from 1 → 8 streams
(the numbers are everywhere, e.g. [Hyper Quen](https://github.com/SYVAI)
measurements).

Volta has working tensor cores and 900 GB/s of HBM2. What it lacks is
`cp.async`, `ldmatrix`, and the modern `mma` shapes — so every fast
quantized-GEMM kernel today requires sm_80+, and Volta is stuck with
leftovers. This repo writes the kernel the other way around: built for
sm_70 first.

## Baseline (measured on the target box, 2026-11-08)

Same server (2× Tesla V100-PCIE-32GB, dual Xeon 8168, 251 GB RAM), same model
(Qwen3.8-27B, UD-Q4_K_M), llama.cpp server, batch = concurrent streams
(128 gen tokens each, greedy, same prompt; combined t/s):

| streams | llama.cpp (CUDA mmq) | reference: single 3090 + vLLM |
|--------:|---------------------:|-------------------------------:|
| 1       | 29                   | 45                             |
| 2       | 46                   | 82                             |
| 4       | 65                   | 154                            |
| 8       | 79                   | 298                            |
| 16      | 99                   | —                              |
| 30      | 112                  | 1035 (at 64)                   |

At batch 1 the weights kernel (`mul_mat_vec_q`) already runs at **74% of DRAM
peak** — single-stream is honest. The batched path is what's broken: it
delivers ~3.8× at 30 streams where Ampere-tuned kernels deliver ~23× at 64.
That gap is kernels, not silicon.

## Target

Batched Q4-K GEMM on sm_70: plain `ld.global` double-buffering into shared
memory (no `cp.async`), register-blocked dequant, `wmma`/`mma` tiles across
concurrent conversations. Reference ceiling on this box ≈ 3500–4000 combined
t/s at batch 32 (short context); a good kernel should clear 1500.

## Layout

- `bench/` — baseline + comparison harnesses
- `kernels/` — CUDA C kernel (`busfill_v0.cu`) + the fragment-layout probe
  (`frag_probe.cu`); `q4k_batched.py` is the Triton attempt, kept for the record
- `RESULTS.md` — running results

## What we learned the hard way

**Triton 2.2 is a dead end for this kernel on sm_70.** Its Volta backend
hard-asserts in MLIR on `tl.trans`, on value reshapes, and on several gather
patterns, and refuses computed `tl.dot` operands unless both sides are fp16 —
and once it finally compiles clean, the result computes *wrong* (max abs err
6.13 against a dequantized matmul on a shape it accepts). Triton 3.x dropped
Volta entirely. The kernel had to be written in raw CUDA with inline PTX.

**`mma.sync.aligned.m8n8k4` works, but the fragment layout must be probed, not
assumed.** The sub-mma group is `s = (lane/4) mod 4` — the high quad *wraps*,
which is exactly the kind of thing a paper-algebra transcription gets wrong in
silence (it computes garbage with no error anywhere). `frag_probe.cu` verifies
A-row, B-column and C-scatter mappings empirically on hardware; both probes
report zero deviation since the fix.

## v0 — raw CUDA, verified, measured (2026-11-08)

K = N = 5120, one V100, batched decode GEMM, dequant fused into the mma loop
(llama.cpp Q4_0 block layout: 32 inputs per block, one fp16 scale).
Correctness gate: max abs err vs dequantized fp16 matmul ≤ 0.6 — passes at
**0.0000** on every batch size. Timing = CUDA events, steady state.

| batch | bus-fill v0 (t/s) | cublas fp16, same shape (t/s) | v0 / cublas |
|------:|------------------:|------------------------------:|------------:|
| 1     | 4,904             | 8,409                         | 58%         |
| 8     | 35,073            | 86,618                        | 40%         |
| 32    | 99,573            | 277,249                       | 36%         |

(Note the cublas reference reads fp16 weights — **4× the bytes** of the
quantized nibble stream — so at batch 32 the quantized kernel is being beaten
by a path that does 4× the memory work. That is the headroom this kernel is
supposed to have.)

Batch scaling on Volta is **real**: 20× from 1 → 32 streams. That is the whole
thesis — the hardware obeys the batching physics; the off-the-shelf kernels
don't. v0 is a first correct implementation, not a tuned one: it reads shared
memory per mma (u16 loads), dequant is SIMT shifts/multiplies, and A-frags
reload per k-pair. Closing on cublas means shared-memory batching (one LDS per
two mmas), the prmt/lop3 dequant trick, and double-buffered `ld.global`
(prefetch next k-block while the tensor cores chew the current one).

## Status

**v1a is the kernel.** Two parallel-agent squads (seven agents), every lever
measured against a control on the same GPU, every variant correctness-gated
(maxerr ≤ 0.6 — all pass at 0.0000). Headline: the dequant bit-twiddle is the
lever — +86% at batch 1, +66% at batch 8, ~21× batch scaling on Volta.
Secondary levers (wider shared loads, deeper prefetch, accumulator
pipelining) all tested negative *on top of* it — see RESULTS.md for both
squads' scoreboards and the mechanism notes. Microbenchmark phase closed;
next is wiring v1a's dequant into a real batched GEMM shape.
