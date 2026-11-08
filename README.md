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
- `kernels/` — the Triton (and, if Triton's Volta backend is too weak, CUDA C) kernels
- `RESULTS.md` — running results

## Status

Spike phase: Triton batched Q4-K GEMM on one V100, batch 1 vs 32, vs the
llama.cpp path. Toolchain pinned to Triton 2.2 / torch 2.2.2 — the last
generation that fully targeted Volta.
