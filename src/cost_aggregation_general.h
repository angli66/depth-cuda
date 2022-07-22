/**
 * By Ang Li, Jul 2022
 */

#ifndef COST_AGGREGATION_GENERAL_H_
#define COST_AGGREGATION_GENERAL_H_

#include <algorithm>

template<class T>
__global__ void CostAggregationKernelUpToDownGeneral(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const int maxDisp, uint8_t* __restrict__ d_disparity) {
    const int x = blockIdx.x; // HxWxD, W position
    const int d = threadIdx.x; // HxWxD, D position

    __shared__ int min;
    extern __shared__ int SharedAggr[]; // length of maxDisp

    // First iteration
    int y = 0;
    const uint8_t cost = d_cost[(y*cols+x)*maxDisp+d];

    int aggr = cost;
    d_L[(y*cols+x)*maxDisp+d] = aggr;

    __syncthreads();
    if (d == 0) { min = 255; }
    __syncthreads();
    atomicMin(&min, aggr);
    SharedAggr[d] = aggr;

    // Remaining iterations
    for (y = 1; y < rows; y++) { // HxWxD, H position
        const uint8_t cost = d_cost[(y*cols+x)*maxDisp+d];
        __syncthreads();

        int left, right;
        if (d != 0) {
            left = SharedAggr[d-1];
        }
        if (d != maxDisp) {
            right = SharedAggr[d+1];
        }
        
        int minimum;
        if (d != 0 && d != maxDisp) {
            minimum = std::min({aggr, left+P1, right+P1, min+P2});
        } else if (d == 0) {
            minimum = std::min({aggr, right+P1, min+P2});
        } else {
            minimum = std::min({aggr, left+P1, min+P2});
        }
        aggr = cost + minimum - min;
        d_L[(y*cols+x)*maxDisp+d] = aggr;

        __syncthreads();
        if (d == 0) { min = 255; }
        __syncthreads();
        atomicMin(&min, aggr);
        SharedAggr[d] = aggr;
    }
}

#endif /* COST_AGGREGATION_GENERAL_H_ */