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
    int in_features = 4096;
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

    // Initialize dummy LUT
    for (int i = 0; i < 65536; ++i) {
        lut[i] = (float)(i % 13) * 0.1f - 0.6f; // Dummy prime distribution
    }

    // Initialize dummy inputs and weights
    for (int i = 0; i < batch_size * in_features; ++i) {
        x[i] = 1.0f; // Easy to verify mathematically
    }
    
    for (int i = 0; i < out_features * in_features; ++i) {
        weight_indices[i] = (uint16_t)(i % 65536);
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
