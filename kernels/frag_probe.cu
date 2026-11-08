// Empirical frag-layout probe for Volta mma.m8n8k4 row.col f16->f32.
// A[32][4] and B[4][8] hold unique small ints so every C element is unique.
// Lanes assemble frags per PTX ISA 8.7 tables; if the layout model is right,
// C == A@B exactly. If not, the mismatch map reveals the true mapping.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

__global__ void probe(const __half* A, const __half* B, float* C) {
    int l = threadIdx.x;
    int q = l & 3, h = l >> 4, s = l >> 2;
    // A frag: (sub-mma s covers rows 8s+q / 8s+q+4) cols 0..3
    uint32_t a[2], b[2];
    // model: a regs = halves {A[row][0],A[row][1]} and {A[row][2],A[row][3]}, row = 8s+q+4h
    int arow = 8 * s + q + 4 * h;
    { __half2 t0 = __halves2half2(A[arow*4+0], A[arow*4+1]); a[0] = *(uint32_t*)&t0;
      __half2 t1 = __halves2half2(A[arow*4+2], A[arow*4+3]); a[1] = *(uint32_t*)&t1; }
    // b regs = halves {B[0][col],B[1][col]} and {B[2][col],B[3][col]}, col = q+4h
    { int col = q + 4 * h;
      __half2 t0 = __halves2half2(B[0*8+col], B[1*8+col]); b[0] = *(uint32_t*)&t0;
      __half2 t1 = __halves2half2(B[2*8+col], B[3*8+col]); b[1] = *(uint32_t*)&t1; }
    float c[8];
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
    // model C-M1: row = 8s + X + 4h, X = (l&1)+(i&2); col = (i&4)+(l&2)+(i&1)
    for (int i = 0; i < 8; i++) {
        int row = 8 * s + (l & 1) + (i & 2) + 4 * h;
        int col = (i & 4) + (l & 2) + (i & 1);
        C[row * 8 + col] = c[i];
    }
}
int main() {
    __half *dA, *dB; float *dC;
    cudaMallocManaged(&dA, 32 * 4 * 2); cudaMallocManaged(&dB, 4 * 8 * 2); cudaMallocManaged(&dC, 32 * 8 * 4);
    for (int i = 0; i < 128; i++) dA[i] = __float2half((float)(i % 61));
    for (int i = 0; i < 32; i++) dB[i] = __float2half((float)(i % 59 + 1));
    probe<<<1, 32>>>(dA, dB, dC); cudaDeviceSynchronize();
    int bad = 0;
    for (int m = 0; m < 32; m++) for (int n = 0; n < 8; n++) {
        float want = 0; for (int k = 0; k < 4; k++) want += __half2float(dA[m * 4 + k]) * __half2float(dB[k * 8 + n]);
        float got = dC[m * 8 + n];
        if (fabsf(got - want) > 0.01f) { if (bad < 10) printf("  C[%d][%d]: got %.1f want %.1f\n", m, n, got, want); bad++; }
    }
    printf("mismatches: %d/256\n", bad);
    return 0;
}
