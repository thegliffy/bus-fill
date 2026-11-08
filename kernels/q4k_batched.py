"""bus-fill v0: batched quant-weight GEMM for sm_70.

The point: weights for ALL batch rows stream from DRAM once per step; the
single memory trip computes B conversations. If aggregate tokens/sec scales
with B on Volta, the bus was idle — the kernel just needs to keep it busy.

Weight layout (Q4-quad, defined by this harness; v1 will eat llama.cpp's
native Q4_K blocks — same math, their 144-byte block format):
  For each output unit (row of W), per BK=128-input chunk (72 bytes):
    - 64 bytes of nibbles: byte i holds inputs (2i, 2i+1), low nibble first,
      value = (nibble - 8) * scale            (symmetric, offset-8 like Q4_0)
    - 4 x fp16 scales, one per 32-input sub-block
  => W bytes = N * (K/128) * 72; handed to the kernel twice from the same
  buffer: a uint8 view for nibbles, a float16 view for scales.

Activations A: [B, K] fp16. Output C: [B, N] fp32 (verified against a
dequantized torch matmul before timing).

On Volta Triton lowers tl.dot through the m16n8k16 fp16 mma path and
software-pipelines plain loads (no cp.async) — exactly the machinery this
kernel exists to measure. Gathers (arange//2 byte index, arange//32 scale
index) keep every load pointer-shaped: Triton 2.2's MLIR pass asserts on
reshaping loaded values.
"""
import torch, time, json, sys
import triton, triton.language as tl
from triton.testing import do_bench


@triton.jit
def q4q_gemm(A, W, Ws, C, Bsz, K: tl.constexpr, N,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             GROUP_M: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < Bsz
    outs = pid_n * BN + tl.arange(0, BN)

    nib_idx = (tl.arange(0, BK) // 2)[None, :]                 # byte per nibble pair
    nib_sh = ((tl.arange(0, BK) % 2) * 4)[None, :]             # low nibble first
    sc_idx = (32 + tl.arange(0, BK) // 32)[None, :]            # fp16 units within chunk
    w_nib = W + outs[:, None] * ((K // 128) * 72) + nib_idx    # u8 ptrs [BN, BK]
    w_sc = Ws + outs[:, None] * ((K // 128) * 36) + sc_idx     # fp16 ptrs [BN, BK]
    a_ptrs = A + rows[:, None] * K + tl.arange(0, BK)[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=rm[:, None], other=0.0)
        b = tl.load(w_nib)                                     # u8 (each byte fetched twice, L1)
        nib = ((b >> nib_sh) & 0xF).to(tl.float16) - 8.0
        sc = tl.load(w_sc)                                     # fp16 (each fetched 32x, L1)
        acc += tl.dot(a, tl.trans(nib * sc))
        a_ptrs += BK
        w_nib += BK // 2
        w_sc += BK // 32
    c_ptrs = C + rows[:, None] * N + outs[None, :]
    tl.store(c_ptrs, acc, mask=rm[:, None])


def quantize_q4q(Wf, device):
    """fp32 [N, K] -> Q4-quad byte buffer + dequantized fp32 reference."""
    N, K = Wf.shape
    w = Wf.to(device).reshape(N, K // 32, 32)
    amax = w.abs().amax(dim=2).clamp(min=1e-6)
    scale = (amax / 8.0)
    q = (w / scale[:, :, None] + 8.0).round().clamp(0, 15).to(torch.uint8)
    lo = q[:, :, 0::2]; hi = q[:, :, 1::2]
    bytes_ = (lo | (hi << 4)).reshape(N, K // 2)
    s16 = scale.half()
    out = torch.empty(N, K // 128 * 72, dtype=torch.uint8, device=device)
    nb = bytes_.reshape(N, K // 128, 64)
    sc = s16.reshape(N, K // 128, 4).view(torch.uint8).reshape(N, K // 128, 8)
    out.view(N, K // 128, 72)[:, :, :64] = nb
    out.view(N, K // 128, 72)[:, :, 64:] = sc
    ref = ((q.float() - 8.0) * s16.float()[:, :, None]).reshape(N, K)
    return out.contiguous(), ref


def launch(A, Wq, C, B, K, N):
    Wq16 = Wq.view(torch.float16)
    grid = (N // 128, 1)
    q4q_gemm[grid](A, Wq, Wq16, C, B, K=K, N=N, BM=32, BN=128, BK=128, GROUP_M=8)


def main():
    dev = "cuda"
    K, N = 5120, 5120
    torch.manual_seed(0)
    Wf = torch.randn(N, K, device=dev) * 0.02
    Aall = torch.randn(32, K, device=dev, dtype=torch.float16)
    Wq, ref = quantize_q4q(Wf, dev)

    # correctness at B=4 (cheap) — fp16 dequant vs fp32 reference matmul
    C = torch.zeros(4, N, device=dev)
    launch(Aall[:4], Wq, C, 4, K, N)
    referr = (C - (Aall[:4].float() @ ref.T)).abs().max().item()
    print(f"max abs err vs dequantized matmul: {referr:.4f}")
    if referr > 0.6:
        print("!! correctness gate failed — timings would be meaningless")
        sys.exit(1)

    results = {"max_abs_err": referr}
    for B in (1, 8, 32):
        C = torch.zeros(B, N, device=dev)
        ms = do_bench(lambda: launch(Aall[:B], Wq, C, B, K, N), warmup=25, rep=100)
        results[B] = round(B / (ms / 1000.0), 1)
        bh = ref.half().contiguous()
        ms_cub = do_bench(lambda: torch.matmul(Aall[:B], bh.T), warmup=25, rep=100)
        results[f"cublas_fp16_{B}"] = round(B / (ms_cub / 1000.0), 1)
        print(f"B={B:2d}: bus-fill {results[B]:9.1f} t/s ({ms*1000:.1f} us)   cublas-fp16 {results[f'cublas_fp16_{B}']:9.1f} t/s ({ms_cub*1000:.1f} us)   ratio {results[B]/results[f'cublas_fp16_{B}']:.3f}")

    json.dump(results, open("busfill_v0.json", "w"), indent=1)


if __name__ == "__main__":
    main()
