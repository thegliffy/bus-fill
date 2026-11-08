import sys, subprocess
if len(sys.argv) > 1:
    # child: compile+bench one config given as BM,BN,BK
    BM, BN, BK, TRANS = map(int, sys.argv[1].split(","))
    import torch, triton, triton.language as tl
    from triton.testing import do_bench
    import os

    @triton.jit
    def probe(A, W, C, Bsz, K: tl.constexpr, N, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TRANS: tl.constexpr):
        pid_n = tl.program_id(0); pid_m = tl.program_id(1)
        rows = pid_m * BM + tl.arange(0, BM)
        outs = pid_n * BN + tl.arange(0, BN)
        a_ptrs = A + rows[:, None] * K + tl.arange(0, BK)[None, :]
        if TRANS:
            w_ptrs = W + outs[:, None] * K + tl.arange(0, BK)[None, :]
        else:
            w_ptrs = W + outs[None, :] + tl.arange(0, BK)[:, None] * N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            w = tl.load(w_ptrs)
            b = tl.trans(w) if TRANS else w
            acc += tl.dot(tl.load(a_ptrs), b)
            a_ptrs += BK
            w_ptrs += (BK if TRANS else BK * N)
        tl.store(C + rows[:, None] * N + outs[None, :], acc)

    dev = "cuda"
    K = N = 5120
    A = torch.randn(32, K, device=dev, dtype=torch.float16)
    Wt = torch.randn(K, N, device=dev, dtype=torch.float16) * 0.02   # [K,N] natural
    Wn = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02   # [N,K] needs trans
    C = torch.zeros(32, N, device=dev)
    W, trans = (Wn, 1) if TRANS else (Wt, 0)
    grid = (N // BN, 32 // BM)
    probe[grid](A, W, C, 32, K=K, N=N, BM=BM, BN=BN, BK=BK, TRANS=trans, num_warps=4)
    torch.cuda.synchronize()
    ms = do_bench(lambda: probe[grid](A, W, C, 32, K=K, N=N, BM=BM, BN=BN, BK=BK, TRANS=trans, num_warps=4), warmup=25, rep=100)
    print(f"BM={BM} BN={BN} BK={BK} trans={trans}: {32/(ms/1000):9.1f} t/s ({ms*1000:.0f} us)")
    sys.exit(0)

for cfg in ["32,32,32", "32,128,128", "16,16,16", "16,64,32", "32,128,64"]:
    for tr in ("0", "1"):
        r = subprocess.run([sys.executable, __file__, cfg + "," + tr], capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        if r.returncode != 0:
            err = (r.stderr.strip().splitlines() or ["?"])[-1]
            line = f"cfg={cfg} trans={tr} CRASH rc={r.returncode} {err[:100]}"
        else:
            line = line
        print(line)
