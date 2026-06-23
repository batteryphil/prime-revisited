/*
 * prime_inference.c
 * Loads the monolithic PRIME Mamba-3 .bin file and executes inference.
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

#define MAGIC_NUM 0x5052494D

typedef struct {
    int magic;
    int d_model;
    int n_layers;
    int vocab_size;
    int lut_size;
} Config;

/* ── kernel declaration ──────────────────────────────────────────── */
void prime_linear_forward(
    const float*    restrict x,
    const uint16_t* restrict weight_indices,
    const float*    restrict lut,
    float*          restrict y,
    int batch_size, int in_features, int out_features);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];

    printf("==========================================================\n");
    printf("  PRIME AVX-512 CPU Inference Engine (Baremetal Loader)\n");
    printf("==========================================================\n");

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    /* Read Header */
    char header[256];
    if (fread(header, 1, 256, f) != 256) {
        fprintf(stderr, "Short read on header\n"); return 1;
    }

    Config* cfg = (Config*)header;
    if (cfg->magic != MAGIC_NUM) {
        fprintf(stderr, "Invalid magic number: 0x%X\n", cfg->magic); return 1;
    }

    printf("[LOAD] Model Config:\n");
    printf("       d_model:    %d\n", cfg->d_model);
    printf("       n_layers:   %d\n", cfg->n_layers);
    printf("       vocab_size: %d\n", cfg->vocab_size);
    printf("       lut_size:   %d\n", cfg->lut_size);

    /* Allocate buffers for layer 0 benchmark */
    float* lut = (float*)malloc(cfg->lut_size * sizeof(float));
    if (fread(lut, sizeof(float), cfg->lut_size, f) != cfg->lut_size) {
        fprintf(stderr, "Short read on LUT\n"); return 1;
    }

    /* Skip embeddings */
    fseek(f, cfg->vocab_size * cfg->d_model * sizeof(float), SEEK_CUR);

    /* Read Layer 0 */
    /* skip norm (2 * d_model * floats) */
    fseek(f, 2 * cfg->d_model * sizeof(float), SEEK_CUR);
    
    /* skip ssm.A_log, ssm.D */
    int d_inner = cfg->d_model * 2; /* from python export: 2048 */
    fseek(f, d_inner * 16 * sizeof(float) + d_inner * sizeof(float), SEEK_CUR);

    /* Read in_proj_idx */
    uint16_t* in_w = (uint16_t*)malloc(d_inner * 2 * cfg->d_model * sizeof(uint16_t));
    if (fread(in_w, sizeof(uint16_t), d_inner * 2 * cfg->d_model, f) != d_inner * 2 * cfg->d_model) {
        fprintf(stderr, "Short read on in_proj\n"); return 1;
    }

    printf("[LOAD] Successfully loaded Layer 0 weights.\n");

    /* Allocate activation buffers */
    float* hidden  = (float*)calloc(cfg->d_model,  sizeof(float));
    float* expand  = (float*)calloc(d_inner * 2,  sizeof(float));

    /* Seed hidden state */
    for (int i = 0; i < cfg->d_model; i++) hidden[i] = 1.0f / cfg->d_model;

    int N_PASSES = 500;
    printf("[INFER] Running baremetal kernel benchmark (%d passes) ...\n", N_PASSES);
    double t0 = now_sec();

    for (int step = 0; step < N_PASSES; step++) {
        prime_linear_forward(hidden, in_w, lut, expand, 1, cfg->d_model, d_inner * 2);
        /* dummy residual to prevent optimization */
        hidden[0] += expand[0] * 0.001f;
    }

    double elapsed = now_sec() - t0;
    double tps     = N_PASSES / elapsed;

    printf("\n----------------------------------------------------------\n");
    printf("  Passes            : %d\n", N_PASSES);
    printf("  Total time        : %.3f s\n",   elapsed);
    printf("  Throughput (TPS)  : %.2f passes/sec\n", tps);
    printf("  ms / pass         : %.3f ms\n",  1000.0 * elapsed / N_PASSES);
    printf("==========================================================\n");

    free(lut); free(in_w); free(hidden); free(expand);
    fclose(f);
    return 0;
}

