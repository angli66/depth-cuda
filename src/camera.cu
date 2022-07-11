/**
 * By Ang Li, Jul 2022
 */

#include "camera.h"

__global__
void disp2Depth(const uint8_t *disp, float *depth,
                    float focalLen, float baselineLen,
                    float minDepth, float maxDepth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (disp[idx] == 0) { // invalidate inf depth
		depth[idx] = 0;
	} else {
    	// depth[idx] = focalLen * baselineLen / disp[idx];
        float temp = focalLen * baselineLen / disp[idx];
        if (temp < minDepth || temp > maxDepth) {
            temp = 0; // invalidate depth out of range
        }
        depth[idx] = temp;
	}
}
