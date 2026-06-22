#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// Declare the kernel function
void prime_linear_forward(
    const float* restrict x,
    const uint16_t* restrict weight_indices,
    const float* restrict lut,
    float* restrict y,
    int batch_size,
    int in_features,
    int out_features
);

// Helper for high-resolution timing
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("==========================================\n");
    printf("PRIME C Inference Kernel Benchmarking\n");
    printf("==========================================\n");

    // Typical LLM sizes (e.g. 1024 d_model)
    int batch_size = 1; // Single token generation latency test
    int in_features = 1024;
    int out_features = 4096;
    
    printf("Config: [%d, %d] x [%d, %d] -> [%d, %d]\n", 
           batch_size, in_features, 
           out_features, in_features, 
           batch_size, out_features);

    // Allocate memory
    float* x = (float*)malloc(batch_size * in_features * sizeof(float));
    uint16_t* weight_indices = (uint16_t*)malloc(out_features * in_features * sizeof(uint16_t));
    float* lut = (float*)malloc(65536 * sizeof(float));
    float* y = (float*)malloc(batch_size * out_features * sizeof(float));

    if (!x || !weight_indices || !lut || !y) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Read real LUT
    FILE* f_lut = fopen("lut.bin", "rb");
    if (!f_lut) { printf("Failed to open lut.bin\n"); return 1; }
    fread(lut, sizeof(float), 65536, f_lut);
    fclose(f_lut);

    // Read real weights
    FILE* f_weights = fopen("weights.bin", "rb");
    if (!f_weights) { printf("Failed to open weights.bin\n"); return 1; }
    fread(weight_indices, sizeof(uint16_t), out_features * in_features, f_weights);
    fclose(f_weights);

    // Initialize dummy inputs (x = 1.0)
    for (int i = 0; i < batch_size * in_features; ++i) {
        x[i] = 1.0f;
    }

    // Warmup run
    prime_linear_forward(x, weight_indices, lut, y, batch_size, in_features, out_features);

    // Benchmark loop
    int num_runs = 1000;
    double start_time = get_time();
    
    for (int r = 0; r < num_runs; ++r) {
        prime_linear_forward(x, weight_indices, lut, y, batch_size, in_features, out_features);
    }
    
    double end_time = get_time();
    double total_time = end_time - start_time;
    double time_per_run_ms = (total_time / num_runs) * 1000.0;
    
    // Check an output to prevent compiler dead-code elimination
    printf("\nSample output y[0]: %f\n", y[0]);
    printf("\n--- Performance Results ---\n");
    printf("Total Time for %d runs: %.3f seconds\n", num_runs, total_time);
    printf("Average Time per pass : %.3f milliseconds\n", time_per_run_ms);
    
    // GFLOPS Calculation
    // Ops per pass: out_features * in_features * 2 (multiply and add)
    double ops_per_pass = 2.0 * out_features * in_features;
    double gflops = (ops_per_pass / (time_per_run_ms / 1000.0)) / 1e9;
    
    printf("Effective throughput  : %.2f GFLOPS\n", gflops);
    printf("==========================================\n");

    free(x);
    free(weight_indices);
    free(lut);
    free(y);

    return 0;
}
