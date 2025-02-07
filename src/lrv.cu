#include <stdio.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda_profiler_api.h>
#include <math.h>
#include <cstdlib>
#include <chrono>

__constant__ const float FLT_MAX = 3.402823466e+38f;  // Maximum finite single-precision float
__constant__ const float FLT_MIN = 1.175494351e-38f;  // Minimum positive normalized single-precision float

// Matrix dimensions and kernel configuration.
const int nb_genes   = 10000; // number of genes (columns)
const int nb_samples = 80;    // number of samples (rows)
const int num_pairs = (nb_genes * (nb_genes - 1)) / 2;

inline __host__ __device__ 
float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__global__
void computeLrvBasic(float* __restrict__ d_Y, 
                    float* __restrict__ d_variances, 
                    int nb_samples, int nb_genes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nb_genes && j < i) {
        // Pack accumulators together to encourage fusion
        float2 accum = make_float2(0.0f, 0.0f);
        int k = 0;

        // Process 4 samples at a time with vector loads
        #pragma unroll
        for (; k <= nb_samples - 4; k += 4) {
            float4 y_i = *reinterpret_cast<float4*>(&d_Y[k + i * nb_samples]);
            float4 y_j = *reinterpret_cast<float4*>(&d_Y[k + j * nb_samples]);

            // Use intrinsics that compiler can fuse
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                // __fdividef has lower precision but can be fused
                float ratio = __fdividef((&y_i.x)[m], (&y_j.x)[m]);
                // __logf can be fused with multiply/add operations
                float log_val = __logf(ratio);

                // Accumulate sum and square together
                accum.x = __fmaf_rn(1.0f, log_val, accum.x); // sum += log_val
                accum.y = __fmaf_rn(log_val, log_val, accum.y); // sumsq += log_val * log_val
            }
        }

        // Handle remaining elements with same fused operations
        for (; k < nb_samples; ++k) {
            float yi = d_Y[k + i * nb_samples];
            float yj = d_Y[k + j * nb_samples];

            float ratio = __fdividef(yi, yj);
            float log_val = __logf(ratio);

            accum.x = __fmaf_rn(1.0f, log_val, accum.x);
            accum.y = __fmaf_rn(log_val, log_val, accum.y);
        }

        // Fused mean and variance computation
        float inv_n = __frcp_rn(static_cast<float>(nb_samples));
        float mean = accum.x * inv_n;
        float variance = (accum.y - __fmul_rn(nb_samples, __fmul_rn(mean, mean))) * __frcp_rn(static_cast<float>(nb_samples - 1));

        int pair_index = (i * (i - 1)) / 2 + j;
        d_variances[pair_index] = variance;
    }
}


__global__
void computeLrvWeighted(float* __restrict__ d_Y, 
                        float* __restrict__ d_W,
                        float* __restrict__ d_variances, 
                        int nb_samples, int nb_genes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nb_genes && j < i) {
        // accum.x = w_sum, accum.y = w_sum2, accum.z = mean, accum.w = numerator (S)
        float4 accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int k = 0;

        // Process 4 samples at a time with vector loads
        #pragma unroll
        for (; k <= nb_samples - 4; k += 4) {
            float4 y_i = *reinterpret_cast<float4*>(&d_Y[k + i * nb_samples]);
            float4 y_j = *reinterpret_cast<float4*>(&d_Y[k + j * nb_samples]);
            float4 w_i = *reinterpret_cast<float4*>(&d_W[k + i * nb_samples]);
            float4 w_j = *reinterpret_cast<float4*>(&d_W[k + j * nb_samples]);
            float4 w_k =  w_i * w_j;

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                float ratio = __fdividef((&y_i.x)[m], (&y_j.x)[m]);
                float log_val = __logf(ratio);
                float mean_old = accum.z;
                float w = (&w_k.x)[m];

                // Update w_sum and w_sum2
                accum.x = __fmaf_rn(1.0f, w, accum.x); // accum.x += w
                accum.y = __fmaf_rn(w, w, accum.y);    // accum.y += w*w

                // Compute new mean
                float delta = log_val - mean_old;
                float w_ratio = __fdividef(w, accum.x);
                accum.z = __fmaf_rn(w_ratio, delta, mean_old);

                // Update numerator (S)
                float delta_new = log_val - accum.z;
                accum.w = __fmaf_rn(w, delta * delta_new, accum.w);
            }
        }

        // Handle remaining elements
        for (; k < nb_samples; ++k) {
            float y_ik = d_Y[k + i * nb_samples];
            float y_jk = d_Y[k + j * nb_samples];
            float w_ik = d_W[k + i * nb_samples];
            float w_jk = d_W[k + j * nb_samples];
            float w_k = w_ik * w_jk;

            float ratio = __fdividef(y_ik, y_jk);
            float log_val = __logf(ratio);
            float mean_old = accum.z;

            // Update w_sum and w_sum2
            accum.x += w_k;
            accum.y += w_k * w_k;

            // Compute new mean
            float delta = log_val - mean_old;
            float w_ratio = w_k / accum.x;
            accum.z += w_ratio * delta;

            // Update numerator (S)
            float delta_new = log_val - accum.z;
            accum.w += w_k * delta * delta_new;
        }

        // Compute lrv
        float S_total = accum.x;
        float Q_total = accum.y;
        float numerator = accum.w;

        float denominator = S_total - (Q_total / S_total);
        float lrv = numerator / denominator;

        int pair_index = (i * (i - 1)) / 2 + j;
        d_variances[pair_index] = lrv;
    }
}

//def online_lrv(data_iterator, a):
//    n = 0
//    mu_i = mu_j = m_i = m_j = 0.0
//    sum_Xi_sq = sum_Xj_sq = 0.0
//    C = 0.0  # Covariance numerator sum
//
//    # Each data point includes X_i, X_j, X_full_i, X_full_j
//    for X_i, X_j, X_full_i, X_full_j in data_iterator:
//        n += 1
//
//        # Update means for X_i and X_j
//        prev_mu_i = mu_i
//        dx_i = X_i - prev_mu_i
//        mu_i += dx_i / n
//
//        prev_mu_j = mu_j
//        dx_j = X_j - prev_mu_j
//        mu_j += dx_j / n
//
//        # Update covariance term using the prior means
//        C += dx_i * (X_j - mu_j)  # X_j's mean was updated after this term
//
//        # Update sums of squares
//        sum_Xi_sq += X_i ** 2
//        sum_Xj_sq += X_j ** 2
//
//        # Update means for X_full_i and X_full_j
//        dx_full_i = X_full_i - m_i
//        m_i += dx_full_i / n
//
//        dx_full_j = X_full_j - m_j
//        m_j += dx_full_j / n
//
//    # Compute adjusted sums for variance terms
//    sum_sq_i = sum_Xi_sq - n * mu_i ** 2
//    sum_sq_j = sum_Xj_sq - n * mu_j ** 2
//
//    # Calculate scaling factors (handle division by zero if needed)
//    a_i = 1.0 / m_i if m_i != 0 else 0.0
//    a_j = 1.0 / m_j if m_j != 0 else 0.0
//
//    # Compute the sum S for LRV
//    S = (sum_sq_i * a_i ** 2) + (sum_sq_j * a_j ** 2) - 2 * a_i * a_j * C
//
//    # Calculate LRV
//    lrv_value = S / (a ** 2 * (n - 1))
//
//    return lrv_value

__global__
void computeLrvAlpha(float* __restrict__ d_Y,
                    float* __restrict__ d_Yfull,
                     float a,
                     float* __restrict__ d_variances,
                     int nb_samples, int nb_genes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nb_genes && j < i) {                    
        float4 mu_m = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  //mu_i = mu_j = m_i = m_j = 0.0
        float2 acc = make_float2(0.0f, 0.0f);               //sum_Xi_sq = sum_Xj_sq = 0.0
        float C = 0.0f;
        int n = 0, k = 0;

        #pragma unroll
        for (; k <= nb_samples - 4; k += 4) {
            float4 y_i     = *reinterpret_cast<float4*>(&d_Y[k + i * nb_samples]);
            float4 y_j     = *reinterpret_cast<float4*>(&d_Y[k + j * nb_samples]);
            float4 yfull_i = *reinterpret_cast<float4*>(&d_Yfull[k + i * nb_samples]);
            float4 yfull_j = *reinterpret_cast<float4*>(&d_Yfull[k + j * nb_samples]);

            // Process each element in the vector
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                n++;
                float inv_n = __frcp_rn(static_cast<float>(n));
                
                // Transform values with power operation
                float X_i      = __powf(reinterpret_cast<float*>(&y_i)[m], a);
                float X_j      = __powf(reinterpret_cast<float*>(&y_j)[m], a);
                float X_full_i = __powf(reinterpret_cast<float*>(&yfull_i)[m], a);
                float X_full_j = __powf(reinterpret_cast<float*>(&yfull_j)[m], a);

                // Update means for X_i and X_j using FMA
                float prev_mu_i = mu_m.x;
                float dx_i = __fsub_rn(X_i, prev_mu_i);
                mu_m.x = __fmaf_rn(dx_i, inv_n, prev_mu_i);

                float prev_mu_j = mu_m.y;
                float dx_j = __fsub_rn(X_j, prev_mu_j);
                mu_m.y = __fmaf_rn(dx_j, inv_n, prev_mu_j);

                // Update covariance term using FMA
                float dxj_muj = __fsub_rn(X_j, mu_m.y);
                C = __fmaf_rn(dx_i, dxj_muj, C);

                // Update sums of squares using FMA
                acc.x = __fmaf_rn(X_i, X_i, acc.x);
                acc.y = __fmaf_rn(X_j, X_j, acc.y);

                // Update means for full data using FMA
                float dx_full_i = __fsub_rn(X_full_i, mu_m.z);
                mu_m.z = __fmaf_rn(dx_full_i, inv_n, mu_m.z);

                float dx_full_j = __fsub_rn(X_full_j, mu_m.w);
                mu_m.w = __fmaf_rn(dx_full_j, inv_n, mu_m.w);
            }
        }

        // Handle remaining samples individually
        for (; k < nb_samples; ++k) {
            n++;
            float inv_n = __frcp_rn(static_cast<float>(n));
            float X_i      = __powf(d_Y[k + i * nb_samples], a);
            float X_j      = __powf(d_Y[k + j * nb_samples], a);
            float X_full_i = __powf(d_Yfull[k + i * nb_samples], a);
            float X_full_j = __powf(d_Yfull[k + j * nb_samples], a);

            // Update means and statistics using FMA
            float prev_mu_i = mu_m.x;
            float dx_i = __fsub_rn(X_i, prev_mu_i);
            mu_m.x = __fmaf_rn(dx_i, inv_n, prev_mu_i);

            float prev_mu_j = mu_m.y;
            float dx_j = __fsub_rn(X_j, prev_mu_j);
            mu_m.y = __fmaf_rn(dx_j, inv_n, prev_mu_j);

            float dxj_muj = __fsub_rn(X_j, mu_m.y);
            C = __fmaf_rn(dx_i, dxj_muj, C);
            
            acc.x = __fmaf_rn(X_i, X_i, acc.x);
            acc.y = __fmaf_rn(X_j, X_j, acc.y);

            float dx_full_i = __fsub_rn(X_full_i, mu_m.z);
            mu_m.z = __fmaf_rn(dx_full_i, inv_n, mu_m.z);

            float dx_full_j = __fsub_rn(X_full_j, mu_m.w);
            mu_m.w = __fmaf_rn(dx_full_j, inv_n, mu_m.w);
        }

        // Compute final LRV value using intrinsics
        float n_mui_sq = __fmul_rn(n, __fmul_rn(mu_m.x, mu_m.x));
        float n_muj_sq = __fmul_rn(n, __fmul_rn(mu_m.y, mu_m.y));
        
        float sum_sq_i = __fsub_rn(acc.x, n_mui_sq);
        float sum_sq_j = __fsub_rn(acc.y, n_muj_sq);

        float a_i = (mu_m.z != 0.0f) ? __frcp_rn(mu_m.z) : 0.0f;
        float a_j = (mu_m.w != 0.0f) ? __frcp_rn(mu_m.w) : 0.0f;

        float ai_sq = __fmul_rn(a_i, a_i);
        float aj_sq = __fmul_rn(a_j, a_j);
        float aij = __fmul_rn(a_i, a_j);

        float term1 = __fmul_rn(sum_sq_i, ai_sq);
        float term2 = __fmul_rn(sum_sq_j, aj_sq);
        float term3 = __fmul_rn(2.0f, __fmul_rn(aij, C));
        
        float S = __fadd_rn(term1, __fsub_rn(term2, term3));
        float a_sq = __fmul_rn(a, a);
        float denom = __fmul_rn(a_sq, static_cast<float>(n - 1));
        float lrv_value = __fdiv_rn(S, denom);

        // Store result
        int pair_index = (i * (i - 1)) / 2 + j;
        d_variances[pair_index] = lrv_value;
    }
}


//def online_lrv_weighted(data_iterator, a):
//    sum_w = sum_w_sq = 0.0
//    sum_wX_i = sum_wX_j = 0.0
//    sum_wX_i_sq = sum_wX_j_sq = 0.0
//    sum_wX_iX_j = 0.0
//    sum_w_full = 0.0
//    sum_w_full_X_full_i = sum_w_full_X_full_j = 0.0
//    # Each data point includes X_i, X_j, X_full_i, X_full_j, W_i, W_j, W_full_i, W_full_j
//    for X_i, X_j, X_full_i, X_full_j, W_i, W_j, W_full_i, W_full_j in data_iterator:
//        w = W_i * W_j
//        w_full = W_full_i * W_full_j
//        # Update sums for the weighted variables
//        sum_w += w
//        sum_w_sq += w ** 2
//        sum_wX_i += w * X_i
//        sum_wX_j += w * X_j
//        sum_wX_i_sq += w * (X_i ** 2)
//        sum_wX_j_sq += w * (X_j ** 2)
//        sum_wX_iX_j += w * X_i * X_j
//        # Update sums for the full variables' weights
//        sum_w_full += w_full
//        sum_w_full_X_full_i += w_full * X_full_i
//        sum_w_full_X_full_j += w_full * X_full_j
//    if sum_w == 0:
//        raise ValueError("Sum of weights cannot be zero.")
//    # Compute means for X_i and X_j using weights
//    mu_i = sum_wX_i / sum_w
//    mu_j = sum_wX_j / sum_w
//    # Compute sums of squares and covariance term
//    sum_sq_i = sum_wX_i_sq - (sum_wX_i ** 2) / sum_w
//    sum_sq_j = sum_wX_j_sq - (sum_wX_j ** 2) / sum_w
//    C = sum_wX_iX_j - (sum_wX_i * sum_wX_j) / sum_w
//    # Compute means for full variables
//    if sum_w_full != 0:
//        m_i = sum_w_full_X_full_i / sum_w_full
//        m_j = sum_w_full_X_full_j / sum_w_full
//    else:
//        m_i = 0.0
//        m_j = 0.0
//    # Handle division by zero for scaling factors
//    a_i = 1.0 / m_i if m_i != 0 else 0.0
//    a_j = 1.0 / m_j if m_j != 0 else 0.0
//    # Compute numerator and denominator for LRV
//    numerator = (sum_sq_i * a_i ** 2) + (sum_sq_j * a_j ** 2) - 2 * a_i * a_j * C
//    denominator_term = sum_w - (sum_w_sq / sum_w)
//    if denominator_term <= 0:
//        raise ValueError("Denominator term must be positive.")
//    denominator = a ** 2 * denominator_term
//    lrv_value = numerator / denominator
//    return lrv_value

__global__
void computeLrvAlphaWeighted(float* __restrict__ d_Y,
                           float* __restrict__ d_Yfull,
                           float* __restrict__ d_W,
                           float* __restrict__ d_Wfull,
                           float a,
                           float* __restrict__ d_variances,
                           int nb_samples, int nb_genes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nb_genes && j < i) {
        // Accumulators for weighted sums
        float sum_w = 0.0f, sum_w_sq = 0.0f;
        float sum_wX_i = 0.0f, sum_wX_j = 0.0f;
        float sum_wX_i_sq = 0.0f, sum_wX_j_sq = 0.0f;
        float sum_wX_iX_j = 0.0f;
        float sum_w_full = 0.0f;
        float sum_w_full_X_full_i = 0.0f, sum_w_full_X_full_j = 0.0f;
        int k = 0;

        // Process 4 samples at a time
        #pragma unroll
        for (; k <= nb_samples - 4; k += 4) {
            float4 y_i     = *reinterpret_cast<float4*>(&d_Y[k + i * nb_samples]);
            float4 y_j     = *reinterpret_cast<float4*>(&d_Y[k + j * nb_samples]);
            float4 yfull_i = *reinterpret_cast<float4*>(&d_Yfull[k + i * nb_samples]);
            float4 yfull_j = *reinterpret_cast<float4*>(&d_Yfull[k + j * nb_samples]);
            float4 w_i     = *reinterpret_cast<float4*>(&d_W[k + i * nb_samples]);
            float4 w_j     = *reinterpret_cast<float4*>(&d_W[k + j * nb_samples]);
            float4 wfull_i = *reinterpret_cast<float4*>(&d_Wfull[k + i * nb_samples]);
            float4 wfull_j = *reinterpret_cast<float4*>(&d_Wfull[k + j * nb_samples]);

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                float X_i      = __powf(reinterpret_cast<float*>(&y_i)[m]    , a);
                float X_j      = __powf(reinterpret_cast<float*>(&y_j)[m]    , a);
                float X_full_i = __powf(reinterpret_cast<float*>(&yfull_i)[m], a);
                float X_full_j = __powf(reinterpret_cast<float*>(&yfull_j)[m], a);
                
                // Compute weights
                float w      = __fmul_rn(reinterpret_cast<float*>(&w_i)[m]    , reinterpret_cast<float*>(&w_j)[m]);
                float w_full = __fmul_rn(reinterpret_cast<float*>(&wfull_i)[m], reinterpret_cast<float*>(&wfull_j)[m]);
                
                // Update weighted sums using FMA
                sum_w = __fadd_rn(sum_w, w);
                sum_w_sq = __fmaf_rn(w, w, sum_w_sq);
                
                sum_wX_i = __fmaf_rn(w, X_i, sum_wX_i);
                sum_wX_j = __fmaf_rn(w, X_j, sum_wX_j);
                
                float X_i_sq = __fmul_rn(X_i, X_i);
                float X_j_sq = __fmul_rn(X_j, X_j);
                sum_wX_i_sq = __fmaf_rn(w, X_i_sq, sum_wX_i_sq);
                sum_wX_j_sq = __fmaf_rn(w, X_j_sq, sum_wX_j_sq);
                
                sum_wX_iX_j = __fmaf_rn(w, __fmul_rn(X_i, X_j), sum_wX_iX_j);
                
                // Update full weight sums
                sum_w_full = __fadd_rn(sum_w_full, w_full);
                sum_w_full_X_full_i = __fmaf_rn(w_full, X_full_i, sum_w_full_X_full_i);
                sum_w_full_X_full_j = __fmaf_rn(w_full, X_full_j, sum_w_full_X_full_j);
            }
        }

        // Handle remaining samples
        for (; k < nb_samples; ++k) {
            float X_i = __powf(d_Y[k + i * nb_samples], a);
            float X_j = __powf(d_Y[k + j * nb_samples], a);
            float X_full_i = d_Yfull[k + i * nb_samples];
            float X_full_j = d_Yfull[k + j * nb_samples];
            
            float w = __fmul_rn(d_W[k + i * nb_samples], d_W[k + j * nb_samples]);
            float w_full = __fmul_rn(d_Wfull[k + i * nb_samples], d_Wfull[k + j * nb_samples]);
            
            sum_w = __fadd_rn(sum_w, w);
            sum_w_sq = __fmaf_rn(w, w, sum_w_sq);
            
            sum_wX_i = __fmaf_rn(w, X_i, sum_wX_i);
            sum_wX_j = __fmaf_rn(w, X_j, sum_wX_j);
            
            float X_i_sq = __fmul_rn(X_i, X_i);
            float X_j_sq = __fmul_rn(X_j, X_j);
            sum_wX_i_sq = __fmaf_rn(w, X_i_sq, sum_wX_i_sq);
            sum_wX_j_sq = __fmaf_rn(w, X_j_sq, sum_wX_j_sq);
            
            sum_wX_iX_j = __fmaf_rn(w, __fmul_rn(X_i, X_j), sum_wX_iX_j);
            
            sum_w_full = __fadd_rn(sum_w_full, w_full);
            sum_w_full_X_full_i = __fmaf_rn(w_full, X_full_i, sum_w_full_X_full_i);
            sum_w_full_X_full_j = __fmaf_rn(w_full, X_full_j, sum_w_full_X_full_j);
        }

        // Compute inverse weights once, using non-zero check
        float inv_sum_w = __frcp_rn(sum_w);
        float inv_sum_w_full = __frcp_rn(sum_w_full);
        
        // Use multiplier that's 0 if sum_w is 0, 1 otherwise
        float w_valid = __fdiv_rn(fminf(fabsf(sum_w), FLT_MAX), __fadd_rn(fabsf(sum_w), FLT_MIN));
        
        // Compute weighted means using predicated multiply
        float mu_i = __fmul_rn(sum_wX_i, __fmul_rn(inv_sum_w, w_valid));
        float mu_j = __fmul_rn(sum_wX_j, __fmul_rn(inv_sum_w, w_valid));
        
        // Compute weighted variances and covariance
        float sum_wX_i_ratio = __fmul_rn(sum_wX_i, sum_wX_i);
        float sum_wX_j_ratio = __fmul_rn(sum_wX_j, sum_wX_j);
        
        float sum_sq_i = __fsub_rn(sum_wX_i_sq, __fmul_rn(sum_wX_i_ratio, inv_sum_w));
        float sum_sq_j = __fsub_rn(sum_wX_j_sq, __fmul_rn(sum_wX_j_ratio, inv_sum_w));
        float C = __fsub_rn(sum_wX_iX_j, __fmul_rn(__fmul_rn(sum_wX_i, sum_wX_j), inv_sum_w));
        
        // Compute means for full variables using predicated multiply
        float wfull_valid = __fdiv_rn(fminf(fabsf(sum_w_full), FLT_MAX), __fadd_rn(fabsf(sum_w_full), FLT_MIN));
        
        float m_i = __fmul_rn(sum_w_full_X_full_i, __fmul_rn(inv_sum_w_full, wfull_valid));
        float m_j = __fmul_rn(sum_w_full_X_full_j, __fmul_rn(inv_sum_w_full, wfull_valid));
        
        // Compute scaling factors using predicated reciprocal
        float m_i_valid = __fdiv_rn(fminf(fabsf(m_i), FLT_MAX), __fadd_rn(fabsf(m_i), FLT_MIN));
        float m_j_valid = __fdiv_rn(fminf(fabsf(m_j), FLT_MAX), __fadd_rn(fabsf(m_j), FLT_MIN));
        
        float a_i = __fmul_rn(__frcp_rn(m_i), m_i_valid);
        float a_j = __fmul_rn(__frcp_rn(m_j), m_j_valid);
        
        float a_i_sq = __fmul_rn(a_i, a_i);
        float a_j_sq = __fmul_rn(a_j, a_j);
        
        // Compute final LRV terms
        float term1 = __fmul_rn(sum_sq_i, a_i_sq);
        float term2 = __fmul_rn(sum_sq_j, a_j_sq);
        float term3 = __fmul_rn(2.0f, __fmul_rn(__fmul_rn(a_i, a_j), C));
        
        float numerator = __fadd_rn(term1, __fsub_rn(term2, term3));
        float denominator_term = __fsub_rn(sum_w, __fmul_rn(sum_w_sq, inv_sum_w));
        
        // Predicate denominator positivity check
        float denom_valid = __fdiv_rn(fminf(denominator_term, FLT_MAX),  __fadd_rn(denominator_term, FLT_MIN));
        float denominator = __fmul_rn(__fmul_rn(a, a), denominator_term);
        float lrv_value   = __fmul_rn(__fdiv_rn(numerator, denominator), denom_valid);
        
        int pair_index = (i * (i - 1)) / 2 + j;
        d_variances[pair_index] = lrv_value;
    }
}


// ---------------------------------------------------------------------
// CPU reference implementation of the full algorithm.
// This version supports both the α–transformation and weighting.
float* compute_log_variance_ratio_cpu_extended(const float* Y, int nb_samples, int nb_genes,
                                                 bool useAlpha, bool useWeighted, float a,
                                                 const float* Yfull, const float* W, const float* Wfull)
{
    int num_pairs = (nb_genes * (nb_genes - 1)) / 2;
    float* variances = new float[num_pairs];
    int counter = 0;

    for (int i = 1; i < nb_genes; i++) {
        for (int j = 0; j < i; j++) {
            if (useAlpha) {
                // α–transformation branch.
                float sum_w = 0.0f, sum_i = 0.0f, sum_j = 0.0f;
                for (int k = 0; k < nb_samples; k++) {
                    float X_i = pow(Y[k + i * nb_samples], a);
                    float X_j = pow(Y[k + j * nb_samples], a);
                    float weight = useWeighted ? (W[k + i * nb_samples] * W[k + j * nb_samples]) : 1.0f;
                    sum_w += weight;
                    sum_i += weight * X_i;
                    sum_j += weight * X_j;
                }
                float mu_i = useWeighted ? (sum_i / sum_w) : (sum_i / nb_samples);
                float mu_j = useWeighted ? (sum_j / sum_w) : (sum_j / nb_samples);

                // Full–data means.
                float sum_full_i = 0.0f, sum_full_j = 0.0f;
                float sum_full_w = 0.0f;
                for (int k = 0; k < nb_samples; k++) {
                    float Xfull_i = pow(Yfull[k + i * nb_samples], a);
                    float Xfull_j = pow(Yfull[k + j * nb_samples], a);
                    float weight = useWeighted ? (Wfull[k + i * nb_samples] * Wfull[k + j * nb_samples]) : 1.0f;
                    sum_full_i += weight * Xfull_i;
                    sum_full_j += weight * Xfull_j;
                    if (useWeighted)
                        sum_full_w += weight;
                }
                float m_i = useWeighted ? (sum_full_i / sum_full_w) : (sum_full_i / nb_samples);
                float m_j = useWeighted ? (sum_full_j / sum_full_w) : (sum_full_j / nb_samples);

                float numerator = 0.0f;
                float S = 0.0f, Q = 0.0f;
                for (int k = 0; k < nb_samples; k++) {
                    float X_i = pow(Y[k + i * nb_samples], a);
                    float X_j = pow(Y[k + j * nb_samples], a);
                    float d_i = (X_i - mu_i) / m_i;
                    float d_j = (X_j - mu_j) / m_j;
                    float diff = d_i - d_j;
                    float weight = useWeighted ? (W[k + i * nb_samples] * W[k + j * nb_samples]) : 1.0f;
                    numerator += weight * diff * diff;
                    if (useWeighted) {
                        S += weight;
                        Q += weight * weight;
                    }
                }
                float denominator = useWeighted ? (a * a * (S - Q / S)) : (a * a * (nb_samples - 1));
                variances[counter++] = numerator / denominator;
            } else {
                // Non–α branch.
                float sum_v = 0.0f;
                if (useWeighted) {
                    float S = 0.0f, Q = 0.0f;
                    for (int k = 0; k < nb_samples; k++) {
                        float v = log(Y[k + i * nb_samples] / Y[k + j * nb_samples]);
                        float weight = W[k + i * nb_samples] * W[k + j * nb_samples];
                        sum_v += weight * v;
                        S += weight;
                        Q += weight * weight;
                    }
                    float mean_v = sum_v / S;
                    float numerator = 0.0f;
                    for (int k = 0; k < nb_samples; k++) {
                        float v = log(Y[k + i * nb_samples] / Y[k + j * nb_samples]);
                        float weight = W[k + i * nb_samples] * W[k + j * nb_samples];
                        float diff = v - mean_v;
                        numerator += weight * diff * diff;
                    }
                    float denominator = S - Q / S;
                    variances[counter++] = numerator / denominator;
                } else {
                    for (int k = 0; k < nb_samples; k++) {
                        float v = log(Y[k + i * nb_samples] / Y[k + j * nb_samples]);
                        sum_v += v;
                    }
                    float mean_v = sum_v / nb_samples;
                    float numerator = 0.0f;
                    for (int k = 0; k < nb_samples; k++) {
                        float v = log(Y[k + i * nb_samples] / Y[k + j * nb_samples]);
                        float diff = v - mean_v;
                        numerator += diff * diff;
                    }
                    variances[counter++] = numerator / (nb_samples - 1);
                }
            }
        }
    }
    return variances;
}

// ---------------------------------------------------------------------
// Performance metrics structure.
struct PerformanceMetrics {
    float kernel_time;   // ms
    float memory_time;   // ms
    float total_time;    // ms
    float gflops;        // GFLOP/s
    float bandwidth;     // GB/s
};

void initializeMatrix(float* Y, int nb_samples, int nb_genes) {
    for (int i = 0; i < nb_samples * nb_genes; i++)
        Y[i] = rand() / (float)RAND_MAX + 1e-3f;
}

// Compare CPU and GPU results.
bool verifyResults(const float* variances_gpu, const float* variances_cpu, int num_pairs) {
    const float epsilon = 1e-2f;
    for (int i = 0; i < num_pairs; i++) {
        if (fabs(variances_gpu[i] - variances_cpu[i]) > epsilon) {
            printf("Verification failed at index %d: GPU=%f, CPU=%f\n", i, variances_gpu[i], variances_cpu[i]);
            return false;
        }
    }
    return true;
}


PerformanceMetrics benchmarkLogVarianceRatio(bool useAlpha, bool useWeighted, float a) {
    PerformanceMetrics metrics;

    size_t size_Y = nb_samples * nb_genes * sizeof(float);
    size_t size_pairs = num_pairs * sizeof(float);
    float *h_Y = (float*)malloc(size_Y);
    float *h_variances_gpu = (float*)malloc(size_pairs);
    float *h_variances_cpu = nullptr;

    initializeMatrix(h_Y, nb_samples, nb_genes);

    // For α–transformation, allocate full–data matrix.
    float *h_Yfull = nullptr;
    if (useAlpha) {
        h_Yfull = (float*)malloc(size_Y);
        for (int i = 0; i < nb_samples * nb_genes; i++)
            h_Yfull[i] = h_Y[i];
    }

    // For weighted branch, allocate weight matrices.
    size_t size_W = size_Y;
    float *h_W = nullptr, *h_Wfull = nullptr;
    if (useWeighted) {
        h_W = (float*)malloc(size_W);
        for (int i = 0; i < nb_samples * nb_genes; i++)
            h_W[i] = rand() / (float)RAND_MAX;
        if (useAlpha) {
            h_Wfull = (float*)malloc(size_W);
            for (int i = 0; i < nb_samples * nb_genes; i++)
                h_Wfull[i] = rand() / (float)RAND_MAX;
        }
    }

    // Allocate device memory.
    float *d_Y, *d_Yfull = nullptr, *d_W = nullptr, *d_Wfull = nullptr, *d_variances;
    cudaMalloc(&d_Y, size_Y);
    if (useAlpha) cudaMalloc(&d_Yfull, size_Y);
    if (useWeighted) {
        cudaMalloc(&d_W, size_W);
        if (useAlpha) cudaMalloc(&d_Wfull, size_W);
    }
    cudaMalloc(&d_variances, size_pairs);

    // Copy data to device.
    // Warmup runs to get GPU into steady state
    const int NUM_WARMUP = 5;
    printf("\nPerforming %d warmup iterations...\n", NUM_WARMUP);
    
    // Do warmup iterations
    for(int i = 0; i < NUM_WARMUP; i++) {
        dim3 blockDim(16, 16);
        dim3 gridDim((nb_genes + blockDim.x - 1) / blockDim.x, (nb_genes + blockDim.y - 1) / blockDim.y);
        // Warmup memory transfers
        cudaMemcpy(d_Y, h_Y, size_Y, cudaMemcpyHostToDevice);
        if (useAlpha) cudaMemcpy(d_Yfull, h_Yfull, size_Y, cudaMemcpyHostToDevice);
        if (useWeighted) {
            cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
            if (useAlpha) cudaMemcpy(d_Wfull, h_Wfull, size_W, cudaMemcpyHostToDevice);
        }

        if (useAlpha) {
            if (useWeighted) {
                computeLrvAlphaWeighted<<<gridDim, blockDim>>>(d_Y, d_Yfull, d_W, d_Wfull, a, d_variances, nb_samples, nb_genes);
            } else {
                computeLrvAlpha<<<gridDim, blockDim>>>(d_Y, d_Yfull, a, d_variances, nb_samples, nb_genes);
            }
        } else {
            if (useWeighted) {
                computeLrvWeighted<<<gridDim, blockDim>>>(d_Y, d_W, d_variances, nb_samples, nb_genes);
            } else {
                computeLrvBasic<<<gridDim, blockDim>>>(d_Y, d_variances, nb_samples, nb_genes);
            }
        }
        
        // Warmup device synchronization and memory transfers back
        cudaMemcpy(h_variances_gpu, d_variances, size_pairs, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    printf("Warmup complete. Starting timed runs...\n\n");

    // Now do the actual timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(d_Y, h_Y, size_Y, cudaMemcpyHostToDevice);
    if (useAlpha) cudaMemcpy(d_Yfull, h_Yfull, size_Y, cudaMemcpyHostToDevice);
    if (useWeighted) {
        cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
        if (useAlpha) cudaMemcpy(d_Wfull, h_Wfull, size_W, cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics.memory_time, start, stop);

    // Launch the appropriate kernel.
    cudaEventRecord(start);
    dim3 blockDim(16, 16);
    dim3 gridDim((nb_genes + blockDim.x - 1) / blockDim.x, (nb_genes + blockDim.y - 1) / blockDim.y);
    if (useAlpha) {
        if (useWeighted) {
            printf("computeLrvAlphaWeighted<<<(%d,%d), (%d,%d)>>>\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
            computeLrvAlphaWeighted<<<gridDim, blockDim>>>(d_Y, d_Yfull, d_W, d_Wfull, a, d_variances, nb_samples, nb_genes);
        } else {
            printf("computeLrvAlpha<<<(%d,%d), (%d,%d)>>>\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
            computeLrvAlpha<<<gridDim, blockDim>>>(d_Y, d_Yfull, a, d_variances, nb_samples, nb_genes);
        }
    } else {
        if (useWeighted) {
            printf("computeLrvWeighted<<<(%d,%d), (%d,%d)>>>\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
            computeLrvWeighted<<<gridDim, blockDim>>>(d_Y, d_W, d_variances, nb_samples, nb_genes);
        } else {
            printf("computeLrvBasic<<<(%d,%d), (%d,%d)>>>\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
            computeLrvBasic<<<gridDim, blockDim>>>(d_Y, d_variances, nb_samples, nb_genes);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics.kernel_time, start, stop);

    // Copy GPU results back.
    cudaMemcpy(h_variances_gpu, d_variances, size_pairs, cudaMemcpyDeviceToHost);

    // Compute CPU version.
    auto cpu_start = std::chrono::high_resolution_clock::now();
    h_variances_cpu = compute_log_variance_ratio_cpu_extended(h_Y, nb_samples, nb_genes,
                                                              useAlpha, useWeighted, a,
                                                              h_Yfull, h_W, h_Wfull);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    // Calculate performance metrics.
    metrics.total_time = metrics.kernel_time + metrics.memory_time;
    // Estimate the FLOP count per gene–pair.
    long long flops_per_pair = 0;
    if (useAlpha) {
        // In the α–branch, each sample does:
        //   – 4 calls to __powf (assume 10 FLOPs each): 40 FLOPs
        //   – Other arithmetic updates (additions, subtractions, FMAs, etc.): ~15 FLOPs
        // Plus an extra ~100 FLOPs for the pair–level (final reductions)
        const int pow_ops  = 40;      // 4 powf’s @ 10 FLOPs each
        const int weight_ops = useWeighted ? 5 : 0;  // extra per–sample if weighted
        const int core_ops = 15;      // arithmetic (add, sub, FMA, etc.)
        flops_per_pair = nb_samples * (pow_ops + weight_ops + core_ops) + 100;
    } else {
        // In the log–ratio branch, each sample does:
        //   – one division + one log (assume 10 FLOPs each → 20 FLOPs)
        //   – Two FMA operations (assume 2 FLOPs each → 4 FLOPs)
        //   → Total per sample ≈ 24 FLOPs
        // Plus an extra ~5 FLOPs for the pair–level (final computations)
        const int log_ops = 20;       // division + log
        const int weight_ops = useWeighted ? 8 : 0;  // extra if weighted
        const int core_ops = 4;       // two FMAs, etc.
        flops_per_pair = nb_samples * (log_ops + weight_ops + core_ops) + 5;
    }

    // Now, total FLOPs is the number of pairs times the FLOPs per pair.
    double total_ops = (double)num_pairs * flops_per_pair;
    metrics.gflops   = total_ops / (metrics.kernel_time * 1e-3 * 1e9);
    float cpu_gflops = (total_ops / 1e9f) / (cpu_time / 1000.0f);
    double bytes_transferred = 0.0;
    bytes_transferred += size_Y;  // Y matrix always
    if (useAlpha) bytes_transferred += size_Y;  // Yfull matrix
    if (useWeighted) {
        bytes_transferred += size_Y;  // W matrix
        if (useAlpha) bytes_transferred += size_Y;  // Wfull matrix
    }
    
    bytes_transferred += size_pairs; 
    bytes_transferred *= 2;
    metrics.bandwidth = bytes_transferred / (metrics.memory_time * 1e6f);  // Convert to GB/s


    bool correct = verifyResults(h_variances_gpu, h_variances_cpu, num_pairs);
    printf("\nResults: %s\n", correct ? "PASSED" : "FAILED");

    printf("\nPerformance Metrics:\n");
    printf("Matrix Size: %dx%d\n", nb_samples, nb_genes);
    printf("Kernel Time:  %.2f ms\n", metrics.kernel_time);
    printf("Memory Time:  %.2f ms\n", metrics.memory_time);
    printf("Total GPU Time: %.2f ms\n", metrics.total_time);
    printf("GPU GFLOPs:   %.2f\n", metrics.gflops);
    printf("GPU Bandwidth:%.2f GB/s\n", metrics.bandwidth);
    printf("\nCPU Time:     %.2f ms\n", cpu_time);
    printf("CPU GFLOPs:   %.2f\n", cpu_gflops);
    printf("Speedup (CPU/GPU): %.2fx\n", cpu_time / metrics.total_time);

    // Cleanup.
    free(h_Y);
    free(h_variances_gpu);
    free(h_variances_cpu);
    if (useAlpha) free(h_Yfull);
    if (useWeighted) {
        free(h_W);
        if (useAlpha) free(h_Wfull);
    }
    cudaFree(d_Y);
    if (useAlpha) cudaFree(d_Yfull);
    if (useWeighted) {
        cudaFree(d_W);
        if (useAlpha) cudaFree(d_Wfull);
    }
    cudaFree(d_variances);

    return metrics;
}

// ---------------------------------------------------------------------
// Main: choose settings and run benchmark.
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    // Test all combinations with nested loops
    const float a = 0.5f;  // α parameter (only used if useAlpha==true)
    bool settings[2] = {false, true};
    
    for (bool useAlpha : settings) {
        for (bool useWeighted : settings) {
            printf("\n=== Log Ratio Variance Benchmark ===\n");
            printf("Settings: useAlpha=%s, useWeighted=%s\n", 
                   useAlpha ? "true" : "false",
                   useWeighted ? "true" : "false");
            if (useAlpha) {
                printf("Alpha value: %.2f\n", a);
            }
            
            PerformanceMetrics metrics = benchmarkLogVarianceRatio(useAlpha, useWeighted, a);
            
            printf("Results:\n");
            printf("  Total GPU Time: %.2f ms\n", metrics.total_time);
            printf("  Kernel Time:    %.2f ms\n", metrics.kernel_time);
            printf("  Memory Time:    %.2f ms\n", metrics.memory_time);
            printf("  GPU GFLOPs:     %.2f\n", metrics.gflops);
            printf("  GPU Bandwidth:  %.2f GB/s\n", metrics.bandwidth);
            printf("----------------------------------------\n");
        }
    }

    return 0;
}
