#include <stdio.h>
#include <stdlib.h>
#include <cmath>
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

__constant__ float avg[32 * 3];

__global__ void kernel(uchar4 *data, int sz, int nc) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    while (idx < sz) {
        uchar4 p = data[idx];
        int jc = 0;
        float fjc = 0.0;
        for (int i = 0; i < nc; i++) {
            float fj = p.x * avg[i * 3] + p.y * avg[i * 3 + 1] + p.z * avg[i * 3 + 2];
            if (fj > fjc) {
                fjc = fj;
                jc = i;
            }
        }
        p.w = (unsigned char)jc;
        data[idx] = p;
        idx += offset;
    }
}

int main() {
    char inputFileName[PATH_MAX];
    scanf("%s", inputFileName);
    char outputFileName[PATH_MAX];
    scanf("%s", outputFileName);

    int w, h;
    FILE *fp = fopen(inputFileName, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    int nc;
    scanf("%d", &nc);
    float *avg_host = (float *)malloc(sizeof(float) * nc * 3);
    for (int i = 0; i < nc; i++) {
        int np;
        scanf("%d", &np);
        int *ps = (int *)malloc(sizeof(int) * np * 2);
        for (int j = 0; j < np * 2; j++) {
            scanf("%d", &ps[j]);
        }
        double sumR = 0.0, sumG = 0.0, sumB = 0.0;
        for (int j = 0; j < np; j++) {
            uchar4 p = data[ps[j * 2 + 1] * w + ps[j * 2]];
            sumR += p.x;
            sumG += p.y;
            sumB += p.z;
        }
        double avgR = sumR / np, avgG = sumG / np, avgB = sumB / np;
        double norm = sqrt(avgR * avgR + avgG * avgG + avgB * avgB);
        avg_host[3 * i] = static_cast<float>(avgR / norm);
        avg_host[3 * i + 1] = static_cast<float>(avgG / norm);
        avg_host[3 * i + 2] = static_cast<float>(avgB / norm);
    }
    
    CSC(cudaMemcpyToSymbol(avg, avg_host, sizeof(float) * nc * 3));

    kernel<<<1024, 1024>>>(dev_data, w * h, nc);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    fp = fopen(outputFileName, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(avg_host);
    cudaFree(dev_data);
    free(data);
    return 0;
}
