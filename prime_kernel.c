#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

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
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int o = 0; o < out_features; ++o) {
            
            const uint16_t* w_row = &weight_indices[o * in_features];
            const float* x_row = &x[b * in_features];
            
            int i = 0;
            float sum = 0.0f;
            
#if defined(__AVX512F__) && defined(__AVX512BW__)
            __m512 sum_vec = _mm512_setzero_ps();
            
            // Unroll by 16 (AVX-512 processes 16 floats at once)
            for (; i <= in_features - 16; i += 16) {
                // Load 16 uint16_t weights (256 bits)
                __m256i w_16bit = _mm256_loadu_si256((const __m256i*)&w_row[i]);
                
                // Zero-extend 16 uint16_t -> 16 uint32_t (512 bits)
                __m512i w_32bit = _mm512_cvtepu16_epi32(w_16bit);
                
                // Hardware GATHER: fetch 16 floats from LUT simultaneously
                __m512 lut_vals = _mm512_i32gather_ps(w_32bit, lut, 4); // scale by 4 bytes (sizeof float)
                
                // Load 16 input floats
                __m512 x_vals = _mm512_loadu_ps(&x_row[i]);
                
                // Fused Multiply-Add
                sum_vec = _mm512_fmadd_ps(lut_vals, x_vals, sum_vec);
            }
            
            // Horizontal sum of the 16 elements in the 512-bit register
            sum += _mm512_reduce_add_ps(sum_vec);
#endif
            
            // Handle tail elements (or fallback if AVX-512 is not compiled)
            for (; i < in_features; ++i) {
                uint16_t idx = w_row[i];
                float w_val = lut[idx];
                sum += x_row[i] * w_val;
            }
            
            y[b * out_features + o] = sum;
        }
    }
}
