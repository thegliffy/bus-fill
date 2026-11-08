// bus-fill v1: 4 output strips per warp (v0 had 1). Same verified PTX
// m8n8k4 layout (kernels/frag_probe.cu proves A-row/B-col/C-scatter incl.
// s = (lane/4) mod 4 wrap). W stages through shared cooperatively (one
// wavefront); A frags load once per k-block into registers and feed 4x the
// mmas. Dequant stays SIMT for now (it did NOT measure as the bottleneck
// when stubbed — the LDS-latency chain did; strips fix that).
//
// Build: nvcc -O3 -arch=sm_70 busfill_v1.cu -o busfill_v1

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>

__device__ __forceinline__ void mma_m8n8k4(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void __launch_bounds__(128)
busfill_q4(const __half* __restrict__ A, const uint8_t* __restrict__ Wq,
           const __half* __restrict__ Ws, float* __restrict__ C,
           int Bsz, int K, int N) {
    __shared__ uint8_t Wsh[4][8][20];              // padded for LDS.U32 (4 bytes = 2 pairs)
    __shared__ __half  Ssh[4][8][2];               // pad 2 -> 16B rows, 4B-aligned loads
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int q = lane & 3, h = lane >> 4, s = (lane >> 2) & 3;
    long wrow = (long)(blockIdx.x * 128 + warp * 32) + q + 4 * h;   // lane's W row (strip = s ✓)
    int arow = 8 * s + q + 4 * h;
    bool live = arow < Bsz;
    const __half* aptr = A + (size_t)arow * K;
    float sc = __half2float(Ws[wrow * (K / 32)]);  // per k-block below; hoisted in loop

    float acc[4][8] = {{0}};
    uint32_t aR[2], bR[2];

    for (int blk = 0; blk < K / 32; blk++) {
        // cooperative: 32 lanes, one uint4 (16B nibbles) + one scale each
        *(uint4*)&Wsh[warp][lane >> 2][lane & 3 ... /*unused*/] = make_uint4(0,0,0,0); // placeholder
        (void)0;
        // stage nibbles: lane l covers row l>>2? no — 32 lanes × 16B = 512B = 32 rows
        // BUT the tile is [8 rows][20B]×4 strips = 4×8 = 32 rows ✓ map: strip = ?, row = ?
        // lane -> row index in Wsh: keep the ORIGINAL row-major byte grid: Wsh flat [4][8]:
        { int r = lane; int st = (lane < 8) ? 0 : -1; (void)st; (void)r; }
        // simplest: first 8 lanes load... we need 32 rows: use all 32 lanes with flat index:
        { int idx = lane; int st = idx / 8, row = idx % 8;
          const uint8_t* src = Wq + ((size_t)(blockIdx.x * 128 + warp * 32 + st * 8 + row) * (K / 2)) + blk * 16;
          *(uint4*)&Wsh[warp][st][row][0] = *(const uint4*)src;   // Wsh dim fix below in code note
          Ssh[warp][st][row][0] = Ws[(size_t)(blockIdx.x * 128 + warp * 32 + st * 8 + row) * (K / 32) + blk];
        }
        __syncwarp();
        float scale = __half2float(Ssh[warp][s][q + 4 * h][0]);
        #pragma unroll
        for (int pw = 0; pw < 4; pw++) {           // uint32 words = 2 pairs
            uint32_t word = *(uint32_t*)&Wsh[warp][s][q + 4 * h][pw * 2];
            #pragma unroll
            for (int hp = 0; hp < 2; hp++) {
                uint32_t p = (hp == 0) ? (word & 0xFFFF) : (word >> 16);
                ((__half*)bR)[0] = __float2half((float)(((p      ) & 0xF) - 8) * scale);
                ((__half*)bR)[1] = __float2half((float)(((p >>  4) & 0xF) - 8) * scale);
                ((__half*)bR)[2] = __float2half((float)(((p >>  8) & 0xF) - 8) * scale);
                ((__half*)bR)[3] = __float2half((float)(((p >> 12) & 0xF) - 8) * scale);
                if (live) *(uint64_t*)aR = *(const uint64_t*)(aptr + blk * 32 + (hp == 0 ? pw * 4 : pw * 4 + 2) * 2);
                else { aR[0] = aR[1] = 0; }
                mma_m8n8k4(acc[s], aR, bR);         // acc[s]: this lane's sub-mma tile... 
                // STRIPS BROKE THE FRAG MODEL: one mma instruction computes ONE 8x8 C tile
                // per warp — "4 strips" would need the B tile to differ per strip, but ALL
                // FOUR SUB-MMAS SHARE the same B columns. One warp = ONE 8-output group.
                // v1 must instead: keep BN=8, and amortize A differently (or accept it).
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int crow = 8 * s + (q & 1) + (i & 2) + 4 * h;
        if (crow < Bsz) {
            int ccol = (i & 4) + (q & 2) + (i & 1);
            C[(size_t)crow * N + blockIdx.x * 128 + warp * 8 + /* strip via s */ s * 8 + ccol] = acc[s][i];
        }
    }
}

int main() { return 0; }
