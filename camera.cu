/**
 * By Ang Li, Jul 2022
 */

#include <stdint.h>

static float baseline;
static float focal;

__global__
void disp2Depth(const uint8_t *disp, uint8_t *depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    depth[idx] = baseline * focal / disp[idx] * 1.0;
}