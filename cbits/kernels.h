#ifndef FUNCTIONAL_ML_KERNELS_H
#define FUNCTIONAL_ML_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

double fml_dot_f64(const double* a, const double* b, int n);

void fml_matvec_bias_f64(const double* W, const double* x, const double* b,
                         double* out, int out_dim, int in_dim);

/*
 * Fused tapeless forward + backward kernel for a 2-layer MLP
 * (in_dim -> hidden_dim [ReLU] -> out_dim [Sigmoid] -> MSE loss)
 * evaluated over a batch slice of `batch_size` samples.
 *
 * params:     [hidden_dim*in_dim + hidden_dim + out_dim*hidden_dim + out_dim]
 * inputs:     [batch_size * in_dim]
 * targets:    [batch_size * out_dim]
 * losses_out: [batch_size]
 * grads_out:  [batch_size * num_params]
 */
void fml_vmap_mlp_value_and_grad_f64(
    const double* params,
    const double* inputs,
    const double* targets,
    double* losses_out,
    double* grads_out,
    int batch_size,
    int in_dim,
    int hidden_dim,
    int out_dim);

#ifdef __cplusplus
}
#endif

#endif
