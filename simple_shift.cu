
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <Windows.h>

#define	MAX(x,y) ((x)>(y)?(x):(y))
#define	MIN(x,y) ((x)<(y)?(x):(y))

cudaError_t addWithCuda(BYTE* a, BYTE* b, unsigned int size, int val);

__global__ void addKernel(BYTE *a, BYTE *b, int max, int val)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= max)
		return;
	auto v = MIN(255, a[pos]+val);
	b[pos] = MAX(0, v);
}

int mainN()
{
	BITMAPFILEHEADER hf;
	BITMAPINFOHEADER hinfo;
	RGBQUAD hRGB[256];
	FILE* fp;
	fp = fopen("l.bmp", "rb");
	if (fp == NULL)
		return -1;
	fread(&hf, sizeof(BITMAPFILEHEADER), 1, fp);
	fread(&hinfo, sizeof(BITMAPINFOHEADER), 1, fp);
	fread(hRGB, sizeof(RGBQUAD), 256, fp);
	int imgSize = hinfo.biWidth * hinfo.biHeight;

	BYTE* image = (BYTE*)malloc(imgSize);
	BYTE* output = (BYTE*)malloc(imgSize);

	fread(image, sizeof(BYTE), imgSize, fp);
	fclose(fp);

	addWithCuda(image, output, imgSize, -40);

	fp = fopen("o.bmp", "wb");
	fwrite(&hf, sizeof(BITMAPFILEHEADER), 1, fp);
	fwrite(&hinfo, sizeof(BITMAPINFOHEADER), 1, fp);
	fwrite(hRGB, sizeof(RGBQUAD), 256, fp);
	fwrite(output, sizeof(BYTE), imgSize, fp);
	fclose(fp);

    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t addWithCuda(BYTE *a, BYTE *b, unsigned int size, int val)
{
	BYTE*dev_a = 0;
	BYTE*dev_b = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(BYTE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(BYTE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(BYTE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	int blocks = floor(size / 1024.0f + 0.5f);
    addKernel<<<blocks, 1024>>>(dev_a, dev_b, size, val);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(b, dev_b, size * sizeof(BYTE), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
