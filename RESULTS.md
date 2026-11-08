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

## Next (v1)

- One shared load per *two* mmas (the u16 read already feeds both nibbles of a
  half-word across two... restructure so each LDS feeds two mma issues).
- prmt/lop3 dequant: build the fp16 mantissa bits directly, multiply by
  1024-fold scale — standard Marlin trick, works fine on Volta.
- Double-buffer shared via `cp.async`'s poor-cousin: plain register-staged
  prefetch of the next k-block while the tensor cores eat this one.
- Target: ≥ 70% of cublas at batch 32 → ~200k t/s aggregate → then wire it
  into llama.cpp's batched path and re-run the ladder.
