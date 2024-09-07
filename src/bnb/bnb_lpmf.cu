#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <helper_cuda.h> //cuda/common/inc

// #include "digamma.hpp"
#include <math_constants.h>


/** 
 * \ingroup prob_dists
 * Computes the digamma function, which is the derivative of the logarithm of the Gamma function, 
 * for a given input `x`.
 *
 * The implementation is inspired by the OpenCL digamma function used in Stan Math Library.
 * https://github.com/stan-dev/math/blob/9b1f08be7e76c6468fc2fcf2f75d641a619ae349/stan/math/opencl/kernels/device_functions/digamma.hpp
 * 
 * @param x The input value for which the digamma function is to be computed.
 * @return The value of the digamma function at `x`.
 * @throw domain_error if `x` is non-positive or causes invalid operations (e.g., division by zero).
 */
__device__ double digamma(double x) {
    double result = 0;
    if (x <= -1) {
        x = 1 - x;
        double remainder = x - floor(x);
        if (remainder > 0.5) {
            remainder -= 1;
        }
        if (remainder == 0) {
            return CUDART_NAN;
        }
        result = CUDART_PI / tan(CUDART_PI * remainder);
    }
    if (x == 0) {
        return CUDART_NAN;
    }
    if (x > 10) {
        const double P[8] = {
            0.083333333333333333333333333333333333333333333333333,
            -0.0083333333333333333333333333333333333333333333333333,
            0.003968253968253968253968253968253968253968253968254,
            -0.0041666666666666666666666666666666666666666666666667,
            0.0075757575757575757575757575757575757575757575757576,
            -0.021092796092796092796092796092796092796092796092796,
            0.083333333333333333333333333333333333333333333333333,
            -0.44325980392156862745098039215686274509803921568627
        };
        x -= 1;
        result += log(x);
        result += 1 / (2 * x);
        double z = 1 / (x * x);
        double tmp = P[7];
        for (int i = 6; i >= 0; i--) {
            tmp = tmp * z + P[i];
        }
        result -= z * tmp;
    } else {
        while (x > 2) {
            x -= 1;
            result += 1 / x;
        }
        while (x < 1) {
            result -= 1 / x;
            x += 1;
        }
        const double Y = 0.99558162689208984;

        const double root1 = 1569415565.0 / 1073741824.0;
        const double root2 = 381566830.0 / 1073741824.0 / 1073741824.0;
        const double root3 = 0.9016312093258695918615325266959189453125e-19;

        const double P[6] = {
            0.25479851061131551,
            -0.32555031186804491,
            -0.65031853770896507,
            -0.28919126444774784,
            -0.045251321448739056,
            -0.0020713321167745952
        };
        const double Q[7] = {
            1.0,
            2.0767117023730469,
            1.4606242909763515,
            0.43593529692665969,
            0.054151797245674225,
            0.0021284987017821144,
            -0.55789841321675513e-6
        };
        double g = x - root1 - root2 - root3;
        double tmp = P[5];
        for (int i = 4; i >= 0; i--) {
            tmp = tmp * (x - 1) + P[i];
        }
        double tmp2 = Q[6];
        for (int i = 5; i >= 0; i--) {
            tmp2 = tmp2 * (x - 1) + Q[i];
        }
        double r = tmp / tmp2;
        result += g * Y + g * r;
    }
    return result;
}




__global__ void bnb_lpmf_kernel(const int* y, const double* r, const double* a, const double* b, double* logp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double one = 1.0;
        logp[i] = lgamma(r[i]+y[i]) + lgamma(b[i]+y[i]) + lgamma(r[i]+a[i]) + lgamma(b[i]+a[i]) - lgamma(y[i]+one) - lgamma(r[i]) - lgamma(b[i]) - lgamma(a[i]) - lgamma(r[i]+a[i]+b[i]+y[i]);
    }
}

__global__ void bnb_deriv_kernel(const int* y, const double* r, const double* a, const double* b, double* p1, double* p2, double* p3, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p1[i] = digamma(y[i]+r[i]) - digamma(r[i]+a[i]+b[i]+y[i]) - digamma(r[i]) + digamma(r[i]+a[i]);
        p2[i] = digamma(a[i]+b[i]) - digamma(r[i]+a[i]+b[i]+y[i]) - digamma(a[i]) + digamma(r[i]+a[i]);
        p3[i] = digamma(a[i]+b[i]) - digamma(r[i]+a[i]+b[i]+y[i]) + digamma(y[i]+b[i]) - digamma(b[i]);
    }
}



// Define pointers to device memory in host code
static int* d_n = nullptr;
static double* d_r = nullptr;
static double* d_a = nullptr;
static double* d_b = nullptr;
static double* d_logp = nullptr;
static double* d_p1 = nullptr;
static double* d_p2 = nullptr;
static double* d_p3 = nullptr;

extern "C" void calculateLpmf(const int* n_data, const double* r_data, const double* a_data, const double* b_data, double* logp, double* p1, double* p2, double* p3, const int vec_size) {
    static int call_count = 0;
    
    size_t size_int = vec_size * sizeof(int);
    size_t size_dbl = vec_size * sizeof(double);

    if (call_count == 0) {
        // Allocate memory on the device
        cudaMalloc((void**)&d_n, size_int);
        cudaMalloc((void**)&d_r, size_dbl);
        cudaMalloc((void**)&d_a, size_dbl);
        cudaMalloc((void**)&d_b, size_dbl);
        cudaMalloc((void**)&d_logp, size_dbl);
        cudaMalloc((void**)&d_p1, size_dbl);
        cudaMalloc((void**)&d_p2, size_dbl);
        cudaMalloc((void**)&d_p3, size_dbl);
    }

    // Copy data to device memory on each call
    cudaMemcpy(d_n, n_data, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r_data, size_dbl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, a_data, size_dbl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data, size_dbl, cudaMemcpyHostToDevice);

    // Launch CUDA kernels
    int threadsPerBlock = 512;
    int blocksPerGrid = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
    bnb_lpmf_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_n, d_r, d_a, d_b, d_logp, vec_size);
    bnb_deriv_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_n, d_r, d_a, d_b, d_p1, d_p2, d_p3, vec_size);

    cudaDeviceSynchronize();

    // Copy results back from device to host
    cudaMemcpy(logp, d_logp, size_dbl, cudaMemcpyDeviceToHost);
    cudaMemcpy(p1, d_p1, size_dbl, cudaMemcpyDeviceToHost);
    cudaMemcpy(p2, d_p2, size_dbl, cudaMemcpyDeviceToHost);
    cudaMemcpy(p3, d_p3, size_dbl, cudaMemcpyDeviceToHost);

    call_count++;
}

