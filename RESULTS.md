# Results

All numbers measured 2026-11-08 on vulcan (Tesla V100-PCIE-32GB, one GPU),
CUDA 12.0 toolkit, `nvcc -O3 -arch=sm_70`. Shape K = N = 5120 (Qwen-scale
hidden layer), batch B = concurrent streams sharing one weight read.

## Correctness gate

Reference = dequantized weights × fp16 activations in torch (float).
Gate: max abs err ≤ 0.6.

| batch | bus-fill v0 max err |
|------:|--------------------:|
| 1     | 0.0000              |
| 8     | 0.0000              |
| 32    | 0.0000              |

## Throughput

CUDA-event timing, steady state, aggregate across all streams.

| batch | bus-fill v0 (t/s) | cublas fp16 (t/s) | v0 / cublas |
|------:|------------------:|------------------:|------------:|
| 1     | 4,904             | 8,409             | 58%         |
| 8     | 35,073            | 86,618            | 40%         |
| 32    | 99,573            | 277,249           | 36%         |

Other cublas reference shapes (fp16 weights):
- K=5120 N=2560: 24,993 / 152,236 / 485,083 t/s at B=1/8/32
- K=2560 N=5120: 15,467 / 132,786 / 488,286 t/s at B=1/8/32

## Fragment layout (mma.m8n8k4 on Volta)

Empirically probed on hardware (`kernels/frag_probe.cu`), both component-wise
probes report **zero deviation** after the fix:

- A row slot per lane: `8*s + q + 4*h`
- B col slot per lane: `q + 4*h`
- C scatter: `row = 8*s + (q&1) + (i&2) + 4*h`, `col = (i&4) + (q&2) + (i&1)`
- where `q = lane%4`, `h = lane/16`, and critically `s = (lane/4) mod 4`
  — the four 8-row groups are *strided by quad within the warp*, not blocked.

Getting this wrong is silent: the kernel runs, no CUDA error, and every number
is garbage.

## Triton 2.2 post-mortem

Triton on sm_70 (last version that targeted Volta at all) could not express
this kernel: hard MLIR assertion on `tl.trans`, hard assertion on value
reshape, gathers fell back to serial, and once everything compiled the kernel
computed max abs err 6.13 against the same reference. Kept in
`kernels/q4k_batched.py` for the record. Raw PTX was required.

## What v0 does

- Warp computes BN=8 outputs × full batch tile (M=32), k-loop over 160 blocks
  of 32 inputs (Q4_0 block size, one fp16 scale per block per output).
- Weights: cooperative 128-bit global load per warp into shared, then
  per-mma `ld.shared.u16` (2 nibbles → 2 mmas-worth... currently 1).
- Dequant: shifts + int→fp16 cvt + scale mul, SIMT, in the mma loop.
- A-frags: loaded from global (L1-hot) per k-pair.
- No double-buffering, no prmt/lop3 trickery, no cp.async (doesn't exist here).

## Engine phase — squad results accumulating (2026-11-08 late eve)

Ladder harness: dense Q4_K_M model, `--parallel 32`, barrier-synchronized
streams × 128 tokens, interleaved arms, file-locked GPU, BFS riddle before
trusting timings.

### L2 (activation-side q8→f16 path): HONEST NULL — no hot loop to patch

Evidence (pinned clone + live nsys profile at n=30): on sm_70, llama.cpp's mmq
dispatch (`use_mma_data_layout()`) excludes Volta — vec-dots run `dp4a` with
activations already packed s8, zero per-value cvt. The q8→half2 cvt chains the
lever targets exist only in Turing+ layouts (dead code here). Where GPU time
actually goes at n=30: **cutlass f16 GEMM 37.5% (prefill → separate
dequantize_block kernels + cuBLAS), weight-side `dequantize_block_*` ~31%,
mul_mat_q dp4a ~4%, f32→f16 unary convert 1.8%.** The harness is dominated by
per-stream prefill, i.e. dequant-to-fp16 + cuBLAS, not by an mma loop. So the
magic-bias lever lives entirely in the weight-side dequantize kernels — which
L1 owns — and L2's assigned path has no cvt mix to remove. Control spot-check:
n=30 → 109.8 t/s, consistent with the recorded 112.

### L1 (weight-side Q4_K dequantize): NO-OP on this path, with SASS evidence

Patched the right thing (`dequantize_q4_K` in `dequantize.cuh` — feeds the
`dequantize_block_q4_K` cuBLAS-prefill path; PRMT→LOP3→HSUB2→HMUL2 per 2
values, exactness documented). Correctness riddle passes identically (answer
`2`, byte-identical outputs). Ladder, 2 interleaved reps per binary, own
control, lock held: Δ at every level between −0.1% and +0.8% = noise. Why:
the patched kernel's hot loop was **already LOP3-folded by ptxas on sm_70**
(control vs patched instruction histograms identical), and the actual
sm_70 GEMM dispatch for Q4_K is the **dp4a family** (28,672 IDP, 0 HMMA in
the cubin — one integer instruction per 4 values, which structurally beats
the fp16 magic-bias chain). The cvt chain the trick targets only exists in
the fp16-tensor-core architecture — which on Volta llama.cpp doesn't take.

### Synthesis so far (L1 + L2, same finding from two directions)

llama.cpp's sm_70 architecture for quantized GEMM is *already* the
instruction-optimal one for Volta's PTX surface: dp4a integer dots amortize
the unpack at 1 instruction/4 values — the Marlin-class idea, natively. The
microbenchmark's magic-bias lever optimizes the fp16-mma architecture, which
real llama.cpp doesn't dispatch here. So end-to-end speedup on Volta cannot
come from patching dequant codegen; it must come from the layers the
microbench never touched: kernel count/launch gaps at batch, scheduling, and
memory-path structure (ncu's original 74%-of-peak at batch 1 stands — the
single-stream kernels are near-roofline; the ladder gap vs Ampere engines is
above the kernel level). This is the answer to "write an engine for them"
part one: the dequant kernel is not the bottleneck at engine level on Volta.

### L3 (Q4_0 data path): pending — last leg.

## v1 squad — four single-lever variants, run in parallel (2026-11-08 pm)

Four agents, one lever each, all gated on maxerr ≤ 0.6 (all passed at 0.0000,
results bitwise-identical where claimed and checked). Control = unmodified v0,
best-of-3 clean runs on the same box. Note: ~40% of runs on this box now die
host-side at CUDA context init (SIGSEGV, apport; hits any binary including the
control — environmental), so every number below is from a complete clean run;
within-binary run-to-run spread < 0.1%.

| variant | lever | B=1 t/s | B=8 t/s | B=32 t/s | vs control B=1/8/32 |
|---------|-------|--------:|--------:|---------:|---------------------|
| control | — (v0) | 4,904   | 35,024  | 99,565   | —                   |
| **v1a** | **prmt/lop3 dequant (8 instr/pair)** | **9,122** | **58,231** | **102,498** | **+86% / +66% / +3%** |
| v1b     | depth-2 register-chain prefetch | 4,990 | 37,837 | 101,681 | +1.8% / +7.9% / +2.1% |
| v1c     | wide LDS (one LDS.U64 feeds 4 mmas) | 5,155 | 38,213 | 104,025 | +5.5% / +9.6% / +4.8% |
| v1d     | 4 output strips/warp, A-frag reuse | 2,294 | 18,324 | 72,783 | **−53% / −47% / −27%** |
| v1      | a+b+c combined (my merge) | 7,785 | 53,713 | 103,884 | +59% / +53% / +4% |

Findings:

- **The dequant instruction count was the real bottleneck** at small/mid
  batch: v1a alone nearly doubles B=1 throughput with the magic-bias
  prmt/lop3 trick (values provably bitwise-identical to v0 — both bias
  halves and sc/16 are exact in fp16).
- **The levers do NOT stack** — the a+b+c combination lands *below* v1a alone
  at B=1/8 (register pressure from the wide loads + chains; ptxas gives up on
  scheduling). Only at B=32 does the combination barely lead, at noise level.
- v1b's dead-on-arrival detail is real science: depth-1 double buffering is
  free because ptxas already pipelines one stage; only depth-2 adds MLP, and
  it adds only ~2-8%.
- v1d's negative result kills the "reuse activation fragments" idea: those
  loads were already L1-hot and latency-hidden; the restructure just spent
  registers (51→72) and lengthened dependency chains.
- ptxas on Volta does NOT fuse `cvt.rn.f16.u8` into the LOP3 pipe — the
  prmt/lop3 path survives at 8 instrs/pair and still wins big.

**Decision: v1a is the kernel.** The path from here is not more single levers —
it's proper shared-memory staging with bank-conflict-free wide frag reads done
in the *dequant's* instruction budget, i.e. restructure around prmt feeding
mma directly, which is where Marlin-class kernels actually live.

## v2 squad — do the levers stack on the dequant fix? All three NO (2026-11-08 eve)

Three agents, each merging ONE lever onto the **v1a base** (the dequant fix),
each running its own same-GPU v1a control built with the same command. All
correctness gates passed at 0.0000; every number is a clean run (v2a: 9+5
clean, v2b: 3+3, v2c: 6+6; ~40% environmental host-side SIGSEGVs as before);
within-binary spread < 0.15%, and sign of the delta consistent across all
clean runs on every variant at every batch.

| variant | lever (on v1a base) | B=1 | B=8 | B=32 | mechanism of the loss |
|---------|---------------------|----:|----:|-----:|------------------------|
| **v1a** | **champion (dequant)** | **9,022–9,122** | **57,424–58,231** | **101,798–102,544** | — |
| v2a | + wide LDS feeding 4 mmas | −5.8% | −6.7% | +1.8% | ptxas can't fold the extraction shifts into PRMT selectors — issue count goes UP |
| v2b | + dual-accumulator interleave | −1.6% | −3.2% | −0.6% | not an occupancy cliff (55 vs 50 regs, same band); just 8 merge FADDs + worse schedule |
| v2c | + depth-2 prefetch chain | −14.2% | −5.8% | +0.1% | not a spill story (47 regs); ptxas schedules the doubled loop worse |

The control at v2b independently reproduced v1a's published numbers, and
v2c's control reproduced them exactly — the negative results sit on
trustworthy baselines.

**What this proves.** The dequant instruction count was the lever, and it has
now been taken once and cannot be taken again: every secondary optimization
the v1 plan proposed (wider shared loads, deeper prefetch, accumulator
pipelining) tested neutral-to-negative **on top of** the cheap dequant. The
loop is ALU-issue-bound on the bit-twiddle itself, and that cost is
structural: the PRMT/LOP3/HSUB2/HMUL2 chain is ~2 instructions per fp16 value,
each value feeds exactly one mma operand slot, and nothing on Volta's
issue/pipe structure amortizes it further. The non-stacking rule now has
evidence in both directions — v1b's mechanism gained on the expensive-dequant
base and loses on the cheap one.

**Where the remaining headroom lives.** At B=32 the kernel streams ~47 GB/s
against a ~900 GB/s bus — nowhere near memory-bound; it's issue-bound on the
dequant. Closing on the cublas fp16 ceiling (which skips dequant entirely by
reading 4× the bytes) requires a fundamentally different dataflow — the
Marlin-class shape where the mma's native width absorbs the unpack — not more
local scheduling. On Volta's PTX surface (m8n8k4 is a quarter-shape) that
restructure may not have a good answer; the squad's verdict is that v1a is
the end of this architecture.

**Decision: v1a is the kernel.** It is ~2× v0 at batch 1, ~1.66× at batch 8,
and its batch-scaling (4,904 → 99,565+ → 102k t/s from B=1 to B=32, ~21×)
confirms the thesis on the hardware. The microbenchmark phase is done; the
next real step is engineering, not scheduling: wire the cheap-dequant path
into a real batched GEMM shape and re-run the llama.cpp ladder for the
end-to-end number.

## Next (v3 / engine)

v2's levers are tested and closed (see v2 squad above — all negative on the
v1a base). Remaining path is structural engineering, not scheduling:

- Wire the v1a dequant into a real batched GEMM shape (the llama.cpp mmq
  structure with the cheap bit-twiddle swapped in) and re-run the ladder —
  that's the end-to-end number the whole exercise exists to produce.
- Only then revisit dataflow (Marlin-class shapes; Volta's PTX surface gives
  the unpack no native width to hide inside, expectations moderate).

