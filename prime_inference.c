/*
 * prime_inference.c
 * Full token-generation inference loop using the AVX-512 PRIME C kernel.
 *
 * Architecture: simulates N token autoregressive generation steps
 * using real PRIME weights (exported from the live checkpoint).
 * Each "token step" runs through:
 *   - in_proj forward  (1024 -> 4096)  [SSM input projection]
 *   - out_proj forward (4096 -> 1024)  [SSM output projection]
 * ...which are the two PRIME layers that dominate inference time.
 *
 * Compile: gcc -O3 -march=native -mavx512f -mavx512bw -mavx512dq
 *               -fopenmp -ffast-math prime_kernel.c prime_inference.c
 *               -o prime_inference -lm
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ── kernel declaration ──────────────────────────────────────────── */
void prime_linear_forward(
    const float*    restrict x,
    const uint16_t* restrict weight_indices,
    const float*    restrict lut,
    float*          restrict y,
    int batch_size, int in_features, int out_features);

/* ── helpers ─────────────────────────────────────────────────────── */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float* load_f32(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* buf = (float*)malloc(n * sizeof(float));
    if (fread(buf, sizeof(float), n, f) != n) {
        fprintf(stderr, "Short read: %s\n", path); exit(1); }
    fclose(f);
    return buf;
}

static uint16_t* load_u16(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    uint16_t* buf = (uint16_t*)malloc(n * sizeof(uint16_t));
    if (fread(buf, sizeof(uint16_t), n, f) != n) {
        fprintf(stderr, "Short read: %s\n", path); exit(1); }
    fclose(f);
    return buf;
}

/* Minimal layer-norm: zero-mean, unit-variance, no affine transform */
static void layer_norm(float* x, int n) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d*d; }
    var  /= n;
    float inv = 1.f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) * inv;
}

/* SiLU activation (used inside Mamba SSM expand) */
static void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        float sig = 1.f / (1.f + expf(-x[i]));
        x[i] *= sig;
    }
}

/* ── main inference loop ──────────────────────────────────────────── */
int main(void) {
    printf("==========================================================\n");
    printf("  PRIME AVX-512 CPU Inference Engine — Step 12500 weights\n");
    printf("==========================================================\n");

    /* Dimensions matching the live training config */
    const int D_MODEL    = 1024;    /* d_model */
    const int D_INNER    = 4096;    /* expand × d_model  (expand=4) */
    const int N_TOKENS   = 512;     /* tokens to generate */
    const int BATCH      = 1;       /* autoregressive single-token */
    const size_t LUT_SZ  = 65536;

    /* Load LUT and in_proj weights from the checkpoint export */
    printf("[LOAD] Reading lut.bin ...\n");
    float*    lut     = load_f32("lut.bin",  LUT_SZ);

    printf("[LOAD] Reading weights.bin (in_proj: %d→%d) ...\n",
           D_MODEL, D_INNER);
    uint16_t* in_w    = load_u16("weights.bin", (size_t)D_INNER * D_MODEL);

    /* For out_proj we reverse-project back; in a real engine this would be
       a separate exported layer.  Here we reuse in_w transposed as a stand-
       in so we exercise exactly the same memory-access pattern.             */
    uint16_t* out_w   = (uint16_t*)malloc((size_t)D_MODEL * D_INNER * sizeof(uint16_t));
    /* Transpose in_w [D_INNER, D_MODEL] -> out_w [D_MODEL, D_INNER] */
    for (int r = 0; r < D_INNER; r++)
        for (int c = 0; c < D_MODEL; c++)
            out_w[(size_t)c * D_INNER + r] = in_w[(size_t)r * D_MODEL + c];

    /* Allocate activation buffers */
    float* hidden  = (float*)calloc(BATCH * D_MODEL,  sizeof(float));
    float* expand  = (float*)calloc(BATCH * D_INNER,  sizeof(float));
    float* output  = (float*)calloc(BATCH * D_MODEL,  sizeof(float));

    /* Seed hidden state with a trivial token embedding (all 1s = BOS) */
    for (int i = 0; i < BATCH * D_MODEL; i++) hidden[i] = 1.0f / D_MODEL;

    printf("[INFER] Generating %d tokens ...\n\n", N_TOKENS);
    double t0 = now_sec();

    for (int step = 0; step < N_TOKENS; step++) {
        /* 1. LayerNorm */
        layer_norm(hidden, D_MODEL);

        /* 2. in_proj: [1, D_MODEL] × [D_INNER, D_MODEL]ᵀ → [1, D_INNER] */
        prime_linear_forward(hidden, in_w, lut, expand, BATCH, D_MODEL, D_INNER);

        /* 3. SiLU nonlinearity (approximates the SSM gate) */
        silu_inplace(expand, D_INNER);

        /* 4. out_proj: [1, D_INNER] × [D_MODEL, D_INNER]ᵀ → [1, D_MODEL] */
        prime_linear_forward(expand, out_w, lut, output, BATCH, D_INNER, D_MODEL);

        /* 5. Residual connection */
        for (int i = 0; i < D_MODEL; i++) hidden[i] += output[i];

        /* 6. Argmax "token sampling" (greedy) */
        int    best_idx = 0;
        float  best_val = hidden[0];
        for (int i = 1; i < D_MODEL; i++)
            if (hidden[i] > best_val) { best_val = hidden[i]; best_idx = i; }

        if (step < 5 || step == N_TOKENS-1)
            printf("  step %4d → token_id %-5d  (logit %.4f)\n",
                   step, best_idx, best_val);
        else if (step == 5)
            printf("  ...\n");
    }

    double elapsed = now_sec() - t0;
    double tps     = N_TOKENS / elapsed;

    printf("\n----------------------------------------------------------\n");
    printf("  Tokens generated  : %d\n", N_TOKENS);
    printf("  Total time        : %.3f s\n",   elapsed);
    printf("  Throughput (TPS)  : %.2f tokens/sec\n", tps);
    printf("  ms / token        : %.3f ms\n",  1000.0 * elapsed / N_TOKENS);
    printf("==========================================================\n");

    free(lut); free(in_w); free(out_w);
    free(hidden); free(expand); free(output);
    return 0;
}
