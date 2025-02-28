#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

struct comparator
{
    __host__ __device__ bool operator()(double a, double b) {
        return a * a < b * b;
    }
};

const double eps = 1e-7;

__global__ void swap_rows(double *matrix, int n, int i, int shift) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(j < n - i) {
        double tmp = matrix[i * n + i + n * j];
        matrix[i * n + i + n * j] = matrix[shift + n * j];
        matrix[shift + n * j] = tmp;
        j += offset;
    }
}

__global__ void gauss(double *matrix, int n, int i) { 
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int h = n - i - 1;
    int w = n - i - 1;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            int k = i + 1 + y;
            int r = i + 1 + x;

            double coef = matrix[k * n + i];
            double num = matrix[i * n + r];
            double div = matrix[i * n + i];
            matrix[k * n + r] -= coef * num / div;
        }
    }
}

int main(int argc, char const *argv[])
{
    int n = 0;
    scanf("%d", &n);
    double *matrix = new double[n * n];
    comparator cmp;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &matrix[n * j + i]);
        }
    }
    
    double *dev_matrix;
    CSC(cudaMalloc(&dev_matrix, sizeof(double) * n * n));
    CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice));

    double det = 1;
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        thrust::device_ptr<double> thrust_matrix = thrust::device_pointer_cast(dev_matrix);
        thrust::device_ptr<double> thrust_max_ptr = thrust::max_element(
            thrust_matrix + (i * n + i), 
            thrust_matrix + (i + 1) * n,
            cmp
        );

        int shift = (int)(thrust_max_ptr - thrust_matrix);

        if (shift != i * n + i) {
            swap_rows<<<1024, 1024>>>(dev_matrix, n, i, shift);
            CSC(cudaGetLastError());
            cnt++;
        }
        
        gauss<<<dim3(64, 64), dim3(16, 16)>>>(dev_matrix, n, i);
        CSC(cudaGetLastError());
    }
    CSC(cudaMemcpy(matrix, dev_matrix, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < n; i++) {
        int shift = i * n + i;
        det *= matrix[shift];
        if (abs(matrix[shift]) < eps) {
            printf("%.10e\n", 0.0);
            CSC(cudaFree(dev_matrix));
            delete[] matrix;
            return 0;
        }
    }
    det *= (-1) * (cnt % 2 == 0 ? -1 : 1);
    printf("%.10e\n", det);

    CSC(cudaFree(dev_matrix));
    delete[] matrix;
    return 0;
}
