#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <Windows.h>
#define	MAX(x,y) ((x)>(y)?(x):(y))
#define	MIN(x,y) ((x)<(y)?(x):(y))
#define XY(x,y,s) (((y)*(s))+(x))

struct Mask {
	int size;
	double* mask;
};

int ApplyMask(const BYTE* img, BYTE* out, const BITMAPINFOHEADER& info, const Mask& mask);

int main()
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
	BYTE * image = (BYTE*)malloc(imgSize);
	BYTE * output = (BYTE*)malloc(imgSize);
	fread(image, sizeof(BYTE), imgSize, fp);
	fclose(fp);


	double r_mask[] =
	{
		0,-1, 0,
		-1, 4, -1,
		0,-1, 0 
	};
	Mask mask{ 3, r_mask };

	ApplyMask(image, output, hinfo, mask);

	fp = fopen("o.bmp", "wb");
	fwrite(&hf, sizeof(BITMAPFILEHEADER), 1, fp);
	fwrite(&hinfo, sizeof(BITMAPINFOHEADER), 1, fp);
	fwrite(hRGB, sizeof(RGBQUAD), 256, fp);
	fwrite(output, sizeof(BYTE), imgSize, fp);
	fclose(fp);

Error:
	free(image);
	free(output);
	fprintf(stdout, "완료");
	cudaError cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

__global__ void kApplyMask(const BYTE* img, const int imgL, BYTE* out, const int maxPos, const double* mask, const int mskL)
{
	int pX = threadIdx.x + blockIdx.x * blockDim.x,
		pY = threadIdx.y + blockIdx.y * blockDim.y;
	int rPos = XY(pX, pY, imgL);
	if (rPos >= maxPos)
		return;

	int M = mskL;
	double sum = 0.0;
	for (int x = 0; x < M; x++)
	{
		for (int y = 0; y < M; y++)
		{
			sum += img[XY(pX + x, pY + y, imgL)] * mask[XY(x, y, mskL)];
		}
	}
	auto v = fabs(sum)/3;
	out[rPos] = (BYTE)MIN(v, 255);
}

int ApplyMask(const BYTE* img, BYTE* out, const BITMAPINFOHEADER& info, const Mask& mask)
{
	cudaError_t cudaStatus;
	BYTE* gp_img, * gp_out;

	if (!(mask.size & 1) || mask.size < 1)
		return 2;
	int size = info.biWidth * info.biHeight;
	if (size < (mask.size * mask.size) || info.biHeight < mask.size || info.biWidth < mask.size)
		return 1;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& gp_img, size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& gp_out, size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gp_img, img, size * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	double* gp_mask;
	cudaStatus = cudaMalloc((void**)& gp_mask, mask.size * mask.size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gp_mask, mask.mask, mask.size * mask.size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int rs = ((info.biBitCount * info.biWidth + 31) / 32 * 4);
	dim3 dblock(floor((info.biWidth - mask.size + 1) / 32.0f + 0.5f), floor((info.biHeight - mask.size + 1) / 32.0f + 0.5f));
	dim3 dthd(32, 32);
	kApplyMask <<<dblock, dthd >>> (gp_img, rs, gp_out, size, gp_mask, mask.size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(out, gp_out, size * sizeof(BYTE), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(gp_img);
	cudaFree(gp_out);
	cudaFree(gp_mask);
	return (cudaStatus == cudaError::cudaSuccess) ? 0 : 3;
}
