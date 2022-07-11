/**
 * By Ang Li, Jul 2022
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#include <stdint.h>

__global__
void disp2Depth(const uint8_t *disp, float *depth,
                    float focalLen, float baselineLen,
                    float minDepth, float maxDepth);

#endif /* CAMERA_H_ */