/**
 * By Ang Li, Jul 2022
 */

#include "cost_aggregation_general.h"

__device__ __forceinline__
int find_minimum(int a1, int a2, int a3, int a4) {
    int minimum = a1;
    if (a2 < minimum) { minimum = a2;}
    if (a3 < minimum) { minimum = a3;}
    if (a4 < minimum) { minimum = a4;}
    return minimum;
}

__global__
void CostAggregationKernelUpToDownGeneral(uint8_t* d_cost, uint8_t *d_L,
                                            const int P1, const int P2,
                                            const int rows, const int cols, const int maxDisp) {
    const int x = blockIdx.x; // HxWxD, W position
    const int d = threadIdx.x; // HxWxD, D position

    __shared__ int min;
    extern __shared__ int SharedAggr[]; // length of maxDisp

    // First iteration
    const uint8_t cost = d_cost[x*maxDisp+d];
    d_L[x*maxDisp+d] = cost;
    if (d == 0) { min = cost; }
    __syncthreads();
    atomicMin(&min, cost);
    SharedAggr[d] = cost;

    // Remaining iterations
    for (int y = 1; y < rows; y++) { // HxWxD, H position
        const uint8_t cost = d_cost[(y*cols+x)*maxDisp+d];
        __syncthreads();

        int left, right;
        if (d != 0) {
            left = SharedAggr[d-1];
        }
        if (d != maxDisp-1) {
            right = SharedAggr[d+1];
        }
        
        int minimum;
        if (d != 0 && d != maxDisp-1) {
            minimum = find_minimum(SharedAggr[d], left+P1, right+P1, min+P2);
        } else if (d == 0) {
            minimum = find_minimum(SharedAggr[d], right+P1, right+P1, min+P2);
        } else {
            minimum = find_minimum(SharedAggr[d], left+P1, left+P1, min+P2);
        }
        int aggr = cost + minimum - min;
        d_L[(y*cols+x)*maxDisp+d] = aggr;

        __syncthreads();
        if (d == 0) { min = aggr; }
        __syncthreads();
        atomicMin(&min, aggr);
        SharedAggr[d] = aggr;
    }
}
