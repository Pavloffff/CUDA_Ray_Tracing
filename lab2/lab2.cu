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


__global__ void kernel(cudaTextureObject_t texObj, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    int x, y;
    uchar4 p;
    int kernelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int kernelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (y = idy; y < h; y += offsety) {
        for (x = idx; x < w; x += offsetx) {
            float Gx = 0.0f, Gy = 0.0f;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    p = tex2D<uchar4>(texObj, x + j, y + i);

                    float YUV = 0.2989f * p.x + 0.587f * p.y + 0.114f * p.z;
                    Gx += YUV * kernelX[i + 1][j + 1];
                    Gy += YUV * kernelY[i + 1][j + 1];
                }
            }

            float f = sqrtf(Gx * Gx + Gy * Gy);
            if (f > 255.0f) {
                f = 255.0f;
            }
            out[y * w + x] = make_uchar4(
                static_cast<u_char>(f), static_cast<u_char>(f), static_cast<u_char>(f), 255
            );
        }
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

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<<dim3(16, 16), dim3(32, 32)>>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(outputFileName, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}
