#!/usr/bin/env python3
"""Roofline probe: how well does cublas batch-scale on sm_70?

If fp16 GEMM at KxN weight shapes scales with batch on Volta, the hardware
can do the work and only the quant kernels are holding us back. Weights
sized like a real 27B layer (hidden 5120, ffn 2560): one weight read per
step should be nearly free at B=32 vs B=1.
"""
import torch, time, json, sys

dev = "cuda"
shapes = [(5120, 5120), (5120, 2560), (2560, 5120)]
results = {}
for K, N in shapes:
    row = {}
    b = torch.randn(K, N, device=dev, dtype=torch.float16)
    for B in (1, 8, 32):
        a = torch.randn(B, K, device=dev, dtype=torch.float16)
        for _ in range(10): c = a @ b
        torch.cuda.synchronize()
        iters = 200
        t0 = time.time()
        for _ in range(iters): c = a @ b
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        row[B] = round(B / dt, 1)
        del a
    results[f"{K}x{N}"] = row
    print(f"K={K} N={N}: " + "  ".join(f"B={k}: {v} t/s" for k, v in row.items()))

json.dump(results, open("baseline_torch.json", "w"), indent=1)

# sanity: memory floor — reading the weight once per step at 900 GB/s
for K, N in shapes[:1]:
    bytes_per_step = K * N * 2
    print(f"memory-floor step time ~{bytes_per_step/900e9*1e6:.1f} us -> ceiling ~{1/(bytes_per_step/900e9):.0f} steps/s")
