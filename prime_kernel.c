#include <stdint.h>
#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Native C inference kernel for the PRIME discrete neural network architecture.
 * 
 * Instead of taking an [out_features, in_features] array of 32-bit floats (which bottleneck memory bandwidth),
 * this takes an array of 16-bit unsigned integers representing LUT indices.
 * It expands the continuous float values on-the-fly directly inside the CPU's L1 cache during the FMA loop.
 *
 * @param x              Input activations matrix [batch_size * in_features]
 * @param weight_indices PRIME compressed weights [out_features * in_features]
 * @param lut            The 65536-element Prime Harmonic Look-Up Table
 * @param y              Output activations matrix [batch_size * out_features]
 * @param batch_size     Number of tokens/sequences
 * @param in_features    Input dimension
 * @param out_features   Output dimension
 */
void prime_linear_forward(
    const float* restrict x,
    const uint16_t* restrict weight_indices,
    const float* restrict lut,
    float* restrict y,
    int batch_size,
    int in_features,
    int out_features
) {
    // For single-token inference (batch_size == 1), parallelizing over out_features is optimal.
    // For large batch training/pre-filling, parallelizing over batch_size might be better,
    // but we use out_features as the outer parallel loop for generation speed.
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            
            // Pointer to the start of this specific output neuron's weights
            const uint16_t* w_row = &weight_indices[o * in_features];
            
            // Pointer to the start of this batch's input activations
            const float* x_row = &x[b * in_features];
            
            // The critical path inner loop.
            // By doing the LUT lookup here, we save 50% memory bandwidth compared to FP32 weights.
            // The compiler will usually auto-vectorize this if AVX2/AVX-512 is enabled.
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < in_features; ++i) {
                uint16_t idx = w_row[i];
                float w_val = lut[idx];
                sum += x_row[i] * w_val;
            }
            
            y[b * out_features + o] = sum;
        }
    }
}
