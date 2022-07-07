/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

	Modified by Ang Li, Jul 2022

**/

#include "disparity_method.h"

static cudaStream_t stream1, stream2, stream3;
static uint8_t *d_im0;
static uint8_t *d_im1;
static cost_t *d_transform0;
static cost_t *d_transform1;
static uint8_t *d_cost;
static uint8_t *d_disparity;
static uint8_t *d_disparity_filtered_uchar;
static uint8_t *h_disparity;
static uint16_t *d_S;
static uint8_t *d_L0;
static uint8_t *d_L1;
static uint8_t *d_L2;
static uint8_t *d_L3;
static uint8_t *d_L4;
static uint8_t *d_L5;
static uint8_t *d_L6;
static uint8_t *d_L7;
static uint8_t p1, p2;
static bool first_alloc;
static uint32_t cols, rows, size, size_cube_l;

void init_disparity_method(const uint8_t _p1, const uint8_t _p2) {
	// Create streams
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
	first_alloc = true;
	p1 = _p1;
	p2 = _p2;
    rows = 0;
    cols = 0;
}

Mat2d<uint8_t> compute_disparity_method(Mat2d<uint8_t> left, Mat2d<uint8_t> right) {
	if(cols != left.cols() || rows != left.rows()) {
		if(!first_alloc) {
			free_memory();
		}
		first_alloc = false;
		cols = left.cols();
		rows = left.rows();
		size = rows*cols;
		size_cube_l = size*MAX_DISPARITY;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));

		int size_cube = size*MAX_DISPARITY;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im0, sizeof(uint8_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im1, sizeof(uint8_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_S, sizeof(uint16_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L0, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L1, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L2, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L3, sizeof(uint8_t)*size_cube_l));
#if PATH_AGGREGATION == 8
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L4, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L5, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L6, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L7, sizeof(uint8_t)*size_cube_l));
#endif

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, sizeof(uint8_t)*size));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*size));
	}
	h_disparity = new uint8_t[size]; // Reset pointer to avoid changing previous result since pybind takes this pointer directly

	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left.data(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right.data(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));

	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (cols+block_size.x-1) / block_size.x;
	grid_size.y = (rows+block_size.y-1) / block_size.y;

	CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	// Hamming distance
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	// Cost Aggregation
	const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

	CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

#if PATH_AGGREGATION == 8
	CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
#endif
	MedianFilter3x3<<<(size+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

	Mat2d<uint8_t> disparity(rows, cols, h_disparity);
	return disparity;
}

static void free_memory() {
	CUDA_CHECK_RETURN(cudaFree(d_im0));
	CUDA_CHECK_RETURN(cudaFree(d_im1));
	CUDA_CHECK_RETURN(cudaFree(d_transform0));
	CUDA_CHECK_RETURN(cudaFree(d_transform1));
	CUDA_CHECK_RETURN(cudaFree(d_L0));
	CUDA_CHECK_RETURN(cudaFree(d_L1));
	CUDA_CHECK_RETURN(cudaFree(d_L2));
	CUDA_CHECK_RETURN(cudaFree(d_L3));
#if PATH_AGGREGATION == 8
	CUDA_CHECK_RETURN(cudaFree(d_L4));
	CUDA_CHECK_RETURN(cudaFree(d_L5));
	CUDA_CHECK_RETURN(cudaFree(d_L6));
	CUDA_CHECK_RETURN(cudaFree(d_L7));
#endif
	CUDA_CHECK_RETURN(cudaFree(d_disparity));
	CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_uchar));
	CUDA_CHECK_RETURN(cudaFree(d_cost));
}

void finish_disparity_method() {
	if(!first_alloc) {
		free_memory();
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream2));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream3));
	}
}