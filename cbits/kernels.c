#include "kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

double fml_dot_f64(const double* a, const double* b, int n) {
#ifdef __aarch64__
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float64x2_t va0 = vld1q_f64(a + i);
        float64x2_t vb0 = vld1q_f64(b + i);
        float64x2_t va1 = vld1q_f64(a + i + 2);
        float64x2_t vb1 = vld1q_f64(b + i + 2);
        acc0 = vfmaq_f64(acc0, va0, vb0);
        acc1 = vfmaq_f64(acc1, va1, vb1);
    }
    float64x2_t sum2 = vaddq_f64(acc0, acc1);
    double sum = vgetq_lane_f64(sum2, 0) + vgetq_lane_f64(sum2, 1);
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

void fml_matvec_bias_f64(const double* W, const double* x, const double* b,
                         double* out, int out_dim, int in_dim) {
    for (int r = 0; r < out_dim; r++) {
        out[r] = fml_dot_f64(W + r * in_dim, x, in_dim) + b[r];
    }
}

void fml_vmap_mlp_value_and_grad_f64(
    const double* __restrict params,
    const double* __restrict inputs,
    const double* __restrict targets,
    double* __restrict losses_out,
    double* __restrict grads_out,
    int batch_size,
    int in_dim,
    int hidden_dim,
    int out_dim) {

    const int l1_w_count = hidden_dim * in_dim;
    const int l1_b_count = hidden_dim;
    const int l2_w_count = out_dim * hidden_dim;
    const int l2_b_count = out_dim;
    const int num_params = l1_w_count + l1_b_count + l2_w_count + l2_b_count;

    const double* w1 = params;
    const double* b1 = w1 + l1_w_count;
    const double* w2 = b1 + l1_b_count;
    const double* b2 = w2 + l2_w_count;

    /* Stack scratchpad for typical hidden/out dims; heap fallback only if huge */
    double stack_buf[512];
    int needed = hidden_dim * 3 + out_dim * 2;
    double* scratch = (needed <= 512) ? stack_buf : (double*)malloc((size_t)needed * sizeof(double));

    double* h_pre  = scratch;
    double* h_act  = h_pre + hidden_dim;
    double* d_h    = h_act + hidden_dim;
    double* o_act  = d_h + hidden_dim;
    double* d_o    = o_act + out_dim;

    const double inv_out = 1.0 / (double)out_dim;

    for (int s = 0; s < batch_size; s++) {
        const double* x = inputs + s * in_dim;
        const double* y = targets + s * out_dim;
        double* g_sample = grads_out + s * num_params;

        double* gw1 = g_sample;
        double* gb1 = gw1 + l1_w_count;
        double* gw2 = gb1 + l1_b_count;
        double* gb2 = gw2 + l2_w_count;

        /* --- Forward: Layer 1 (Linear + ReLU) --- */
        for (int h = 0; h < hidden_dim; h++) {
            double z = fml_dot_f64(w1 + h * in_dim, x, in_dim) + b1[h];
            h_pre[h] = z;
            h_act[h] = (z > 0.0) ? z : 0.0;
        }

        /* --- Forward: Layer 2 (Linear + Sigmoid) + MSE Loss --- */
        double loss = 0.0;
        for (int o = 0; o < out_dim; o++) {
            double z = fml_dot_f64(w2 + o * hidden_dim, h_act, hidden_dim) + b2[o];
            double pred = 1.0 / (1.0 + exp(-z));
            o_act[o] = pred;
            double diff = pred - y[o];
            loss += diff * diff;
            /* d(loss)/d(z_o) = (2/out_dim) * (pred - y) * pred * (1 - pred) */
            d_o[o] = (2.0 * inv_out) * diff * pred * (1.0 - pred);
        }
        losses_out[s] = loss * inv_out;

        /* --- Backward: Layer 2 --- */
        memset(d_h, 0, (size_t)hidden_dim * sizeof(double));
        for (int o = 0; o < out_dim; o++) {
            double go = d_o[o];
            gb2[o] = go;
            const double* w2_row = w2 + o * hidden_dim;
            double* gw2_row = gw2 + o * hidden_dim;
#ifdef __aarch64__
            float64x2_t vgo = vdupq_n_f64(go);
            int h = 0;
            for (; h + 2 <= hidden_dim; h += 2) {
                float64x2_t vh = vld1q_f64(h_act + h);
                float64x2_t vw = vld1q_f64(w2_row + h);
                float64x2_t vdh = vld1q_f64(d_h + h);
                vst1q_f64(gw2_row + h, vmulq_f64(vgo, vh));
                vst1q_f64(d_h + h, vfmaq_f64(vdh, vgo, vw));
            }
            for (; h < hidden_dim; h++) {
                gw2_row[h] = go * h_act[h];
                d_h[h] += go * w2_row[h];
            }
#else
            for (int h = 0; h < hidden_dim; h++) {
                gw2_row[h] = go * h_act[h];
                d_h[h] += go * w2_row[h];
            }
#endif
        }

        /* --- Backward: Layer 1 (ReLU + Linear) --- */
        for (int h = 0; h < hidden_dim; h++) {
            double gh = (h_pre[h] > 0.0) ? d_h[h] : 0.0;
            gb1[h] = gh;
            double* gw1_row = gw1 + h * in_dim;
#ifdef __aarch64__
            float64x2_t vgh = vdupq_n_f64(gh);
            int i = 0;
            for (; i + 2 <= in_dim; i += 2) {
                float64x2_t vx = vld1q_f64(x + i);
                vst1q_f64(gw1_row + i, vmulq_f64(vgh, vx));
            }
            for (; i < in_dim; i++) {
                gw1_row[i] = gh * x[i];
            }
#else
            for (int i = 0; i < in_dim; i++) {
                gw1_row[i] = gh * x[i];
            }
#endif
        }
    }

    if (scratch != stack_buf) {
        free(scratch);
    }
}
