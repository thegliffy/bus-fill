"""bus-fill v0: batched quant-weight GEMM for sm_70.

The point: weights for ALL batch rows stream from DRAM once per step; the
single memory trip computes B conversations. If aggregate tokens/sec scales
with B on Volta, the bus was idle — the kernel just needs to keep it busy.

Weight layout (Q4-quad, defined by this kernel; v1 will eat llama.cpp's
native Q4_K blocks — same math, their 144-byte block format):
  For each output unit (row of W), per BK=128-input chunk:
    - 64 bytes of nibbles: byte i holds inputs (2i, 2i+1), low nibble first,
      value = (nibble - 8) * scale            (symmetric, offset-8 like Q4_0)
    - 4 x fp16 scales, one per 32-input sub-block
  => W bytes = N * (K/128) * (64 + 8)

Activations A: [B, K] fp16, B padded to BM=32 (tl.dot minimum tile).
Output C: [B, N] fp32 (verify against dequantized torch matmul).

On Volta Triton lowers tl.dot through the m16n8k16 fp16 mma path and
software-pipelines plain loads (no cp.async) — which is exactly the
machinery this kernel is here to measure.
"""
import torch, time, json, sys
import triton, triton.language as tl
from triton.testing import do_bench


@triton.jit
def q4q_gemm(A, W, C, Bsz, K: tl.constexpr, N,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             GROUP_M: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < Bsz
    outs = pid_n * BN + tl.arange(0, BN)

    # W per output row: K/BK chunks; each chunk = 64B nibbles then 8B (4x fp16) scales
    w_base = W + (outs[:, None] * (K // BK) * 72)             # [BN, 1] bytes
    nib_off = tl.arange(0, BK)[None, :] // 2                   # byte per nibble pair (loaded twice, L1-fed)
    nib_sh = ((tl.arange(0, BK) % 2) * 4)[None, :]             # low nibble first
    sc_off = (BK // 2 + tl.arange(0, BK // 32)[None, :] * 2)   # fp16 via byte offset
    a_ptrs = A + rows[:, None] * K + tl.arange(0, BK)[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=rm[:, None], other=0.0)                       # [BM, BK] fp16
        b = tl.load(w_base + k0 // BK * 72 + nib_off)                           # [BN, BK] u8 (each byte twice)
        nib = ((b >> nib_sh) & 0xF).to(tl.float32) - 8.0                         # [BN, BK]
        sc = tl.load((w_base + k0 // BK * 72 + sc_off).to(tl.pointer_type(tl.float16)))  # [BN, BK/32] fp16
        w = (nib.reshape(BN, BK // 32, 32) * sc[:, :, None]).reshape(BN, BK)     # dequantized
        acc += tl.dot(a, w.trans())
        a_ptrs += BK
    c_ptrs = C + rows[:, None] * N + outs[None, :]
    tl.store(c_ptrs, acc, mask=rm[:, None])


def quantize_q4q(Wf, device):
    """fp32 [N, K] -> our Q4-quad layout, for the harness + reference."""
    N, K = Wf.shape
    w = Wf.to(device).reshape(N, K // 32, 32)
    amax = w.abs().amax(dim=2).clamp(min=1e-6)
    scale = (amax / 8.0)                                   # symmetric, offset-8
    q = (w / scale[:, :, None] + 8.0).round().clamp(0, 15).to(torch.uint8)
    lo = q[:, :, 0::2]; hi = q[:, :, 1::2]
    bytes_ = (lo | (hi << 4)).reshape(N, K // 2)            # byte i covers inputs 2i,2i+1
    s16 = scale.half()
    out = torch.empty(N, K // 128 * 72, dtype=torch.uint8, device=device)
    # per 128-input chunk: 64 bytes nibbles + 8 bytes scales
    nb = bytes_.reshape(N, K // 128, 64)
    sc = s16.reshape(N, K // 128, 4).view(torch.uint8).reshape(N, K // 128, 8)
    out.view(N, K // 128, 72)[:, :, :64] = nb
    out.view(N, K // 128, 72)[:, :, 64:] = sc
    # dequant reference (exact, in fp32, for correctness)
    ref = ((q.float() - 8.0) * s16.float()[:, :, None]).reshape(N, K)
    return out.contiguous(), ref


def main():
    dev = "cuda"
    K, N = 5120, 5120
    torch.manual_seed(0)
    Wf = torch.randn(N, K, device=dev) * 0.02
    Aall = torch.randn(32, K, device=dev)
    Wq, ref = quantize_q4q(Wf, dev)

    # correctness at B=4 (cheap)
    C = torch.zeros(4, N, device=dev)
    q4q_gemm[(N // 128, 1)](Aall[:4], Wq, C, 4, K=K, N=N, BM=32, BN=128, BK=128, GROUP_M=8)
    referr = (C - (Aall[:4] @ ref.T)).abs().max().item()
    print(f"max abs err vs dequantized matmul: {referr:.4f}")

    results = {}
    for B in (1, 8, 32):
        C = torch.zeros(B, N, device=dev)
        grid = (N // 128, 1)
        args = (Aall[:B], Wq, C, B)
        kw = dict(K=K, N=N, BM=32, BN=128, BK=128, GROUP_M=8)
        ms = do_bench(lambda: q4q_gemm[grid](*args, **kw), warmup=25, rep=100)
        results[B] = round(B / (ms / 1000.0), 1)
        # cublas fp16 reference at same shape, same methodology
        bh = Wf.half()
        ms_cub = do_bench(lambda: torch.matmul(Aall[:B], bh.T), warmup=25, rep=100)
        results[f"cublas_fp16_{B}"] = round(B / (ms_cub / 1000.0), 1)
        print(f"B={B:2d}: bus-fill {results[B]:9.1f} t/s ({ms*1000:.1f} us)   cublas-fp16 {results[f'cublas_fp16_{B}']:9.1f} t/s ({ms_cub*1000:.1f} us)   ratio {results[B]/results[f'cublas_fp16_{B}']:.3f}")

    json.dump(results, open("busfill_v0.json", "w"), indent=1)


if __name__ == "__main__":
    main()
