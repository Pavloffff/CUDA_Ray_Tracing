#include <new>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void print_checker_test(int n, double *vec)
{
    fprintf(stderr, "%d\n", n);
    if (n > 1999) {
        n = 1999;
    }
    if (vec != NULL) {
        for (int i = 0; i < n; i++) {
            fprintf(stderr, "%lf", vec[i]);
            if (i < n - 1) fprintf(stderr, " ");
        }
    }
}

__global__ void kernel(double *inp_vec, double *ans_vec, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    while (idx < n) {
        ans_vec[n - idx - 1] = inp_vec[idx];
        idx += offset;
    }
}

int main(int argc, char const *argv[])
{
    int n = 0;
    if (scanf("%d", &n) == -1) {
        fprintf(stderr, "ERROR: Failed to read n\n");
        return 0;
    }
    
    double *host_vec = new(std::nothrow) double[n];
    if (host_vec == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate host_vec\n");
        return 0;
    }
    
    for (int i = 0; i < n; i++) {
        if (scanf("%lf", &host_vec[i]) == -1) {
            fprintf(stderr, "ERROR: Failed to read host_vec[%d]\n", i);
            print_checker_test(n, host_vec);
            delete[] host_vec;
            return 0;
        }
    }

    double *dev_inpvec;
    cudaError_t err;

    err = cudaMalloc(&dev_inpvec, sizeof(double) * n);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc failed for dev_inpvec: %s\n", cudaGetErrorString(err));
        print_checker_test(n, host_vec);
        delete[] host_vec;
        return 0;
    }

    err = cudaMemcpy(dev_inpvec, host_vec, sizeof(double) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy HostToDevice failed for dev_inpvec: %s\n", cudaGetErrorString(err));
        print_checker_test(n, host_vec);
        cudaFree(dev_inpvec);
        delete[] host_vec;
        return 0;
    }
    
    double *dev_ansvec;
    err = cudaMalloc(&dev_ansvec, sizeof(double) * n);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc failed for dev_ansvec: %s\n", cudaGetErrorString(err));
        print_checker_test(n, host_vec);
        cudaFree(dev_inpvec);
        delete[] host_vec;
        return 0;
    }
    
    kernel<<<1024, 1024>>>(dev_inpvec, dev_ansvec, n);
    
    err = cudaMemcpy(host_vec, dev_ansvec, sizeof(double) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy DeviceToHost failed for dev_inpvec: %s\n", cudaGetErrorString(err));
        print_checker_test(n, host_vec);
        cudaFree(dev_inpvec);
        cudaFree(dev_ansvec);
        delete[] host_vec;
        return 0;
    }

    for (int i = 0; i < n; i++) {
        printf("%.10e ", host_vec[i]);
    }
    printf("\n");

    cudaFree(dev_inpvec);
    cudaFree(dev_ansvec);
    delete[] host_vec;
    return 0;
}
