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

#include "depth_method.h"

static cudaStream_t stream1, stream2, stream3;
static uint8_t *d_rawim0;
static uint8_t *d_rawim1;
static uint8_t *d_im0;
static uint8_t *d_im1;
static cost_t *d_transform0;
static cost_t *d_transform1;
static uint8_t *d_cost;
static uint8_t *d_disparity;
static uint8_t *d_disparity_filtered_uchar;
static float *d_depth;
static float *h_depth;
static uint8_t *d_L0;
static uint8_t *d_L1;
static uint8_t *d_L2;
static uint8_t *d_L3;
static uint8_t *d_L4;
static uint8_t *d_L5;
static uint8_t *d_L6;
static uint8_t *d_L7;
static uint8_t p1, p2;
static bool memory_occupied;
static uint32_t cols, rows, size, size_cube_l;
static float focalLen, baselineLen, minDepth, maxDepth;
static bool rectified;
static uint8_t censusWidth, censusHeight;
static float *d_mapLx;
static float *d_mapLy;
static float *d_mapRx;
static float *d_mapRy;

void init_depth_method(const uint8_t _p1, const uint8_t _p2, uint32_t _cols, uint32_t _rows,
						float _focalLen, float _baselineLen, float _minDepth, float _maxDepth,
						Mat2d<float> mapLx, Mat2d<float> mapLy, Mat2d<float> mapRx, Mat2d<float> mapRy, bool _rectified,
						uint8_t _censusWidth, uint8_t _censusHeight) {
	// Create streams and free memory if necessary
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
	if(memory_occupied) { free_memory(); }

	// Intialize global variables
	p1 = _p1;
	p2 = _p2;
	cols = _cols;
    rows = _rows;
	focalLen = _focalLen;
	baselineLen = _baselineLen;
	minDepth = _minDepth;
	maxDepth = _maxDepth;
	rectified = _rectified;
	censusWidth = _censusWidth;
	censusHeight = _censusHeight;

	size = rows*cols;
	size_cube_l = size*MAX_DISPARITY;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));
	int size_cube = size*MAX_DISPARITY;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube));
	if (!rectified) {
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_rawim0, sizeof(uint8_t)*size));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_rawim1, sizeof(uint8_t)*size));
	}
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im0, sizeof(uint8_t)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im1, sizeof(uint8_t)*size));
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
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_depth, sizeof(float)*size));
	memory_occupied = true;

	// Allocate inverse maps on GPU
	size = rows*cols;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_mapLx, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_mapLy, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_mapRx, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_mapRy, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_mapLx, mapLx.data(), sizeof(float)*size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_mapLy, mapLy.data(), sizeof(float)*size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_mapRx, mapRx.data(), sizeof(float)*size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_mapRy, mapRy.data(), sizeof(float)*size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

Mat2d<float> compute_depth_method(Mat2d<uint8_t> left, Mat2d<uint8_t> right) {
	if (cols != left.cols() || rows != left.rows()) { throw std::runtime_error("Input image size different from initiated"); }
	h_depth = new float[size]; // Reset pointer to avoid changing previous result since pybind takes this pointer directly

	if (rectified) {
		CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left.data(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right.data(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice));
	} else {
		CUDA_CHECK_RETURN(cudaMemcpyAsync(d_rawim0, left.data(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpyAsync(d_rawim1, right.data(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice));
		dim3 bs;
		bs.x = 32*32;
		dim3 gs;
		gs.x = (size+bs.x-1) / bs.x;
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		remap<<<gs, bs, 0, stream1>>>(d_mapLx, d_mapLy, rows, cols, d_rawim0, d_im0);
		remap<<<gs, bs, 0, stream2>>>(d_mapRx, d_mapRy, rows, cols, d_rawim1, d_im1);
	}

	dim3 bs2;
	bs2.x = 32;
	bs2.y = 32;
	dim3 gs2;
	gs2.x = (cols+bs2.x-1) / bs2.x;
	gs2.y = (rows+bs2.y-1) / bs2.y;
	const unsigned int win_cols = (31+censusWidth);
	const unsigned int win_rows = (31+censusHeight);
	const unsigned int shared_mem_size = 2 * win_cols * win_rows * sizeof(uint8_t);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CenterSymmetricCensusKernelSM2<<<gs2, bs2, shared_mem_size, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols, censusWidth, censusHeight);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	// Hamming distance
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
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

	// Disparity to depth conversion
	dim3 bs3;
	bs3.x = 32*32;
	dim3 gs3;
	gs3.x = (size+bs3.x-1) / bs3.x;
	disp2Depth<<<gs3, bs3, 0, stream1>>>(d_disparity_filtered_uchar, d_depth, focalLen, baselineLen, minDepth, maxDepth);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(h_depth, d_depth, sizeof(float)*size, cudaMemcpyDeviceToHost));

	Mat2d<float> depth(rows, cols, h_depth);
	return depth;
}

static void free_memory() {
	if (!rectified) {
		CUDA_CHECK_RETURN(cudaFree(d_rawim0));
		CUDA_CHECK_RETURN(cudaFree(d_rawim1));
	}
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
	CUDA_CHECK_RETURN(cudaFree(d_mapLx));
	CUDA_CHECK_RETURN(cudaFree(d_mapLy));
	CUDA_CHECK_RETURN(cudaFree(d_mapRx));
	CUDA_CHECK_RETURN(cudaFree(d_mapRy));
	memory_occupied = false;
}

void finish_depth_method() {
	free_memory();
	CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
	CUDA_CHECK_RETURN(cudaStreamDestroy(stream2));
	CUDA_CHECK_RETURN(cudaStreamDestroy(stream3));
}
