/**
 * By Ang Li, Jul 2022
 */

#include "camera.h"

__global__
void remap(const float *mapx, const float *mapy,
            const uint32_t rows, const uint32_t cols,
            const uint8_t *src, uint8_t *dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sx = mapx[idx];
    float sy = mapy[idx];

    // Eliminate error
    int x, y;
    bool xdone = false;
    bool ydone = false;
    if (sx - round(sx) <= 0.01 || sx - round(sx) <= -0.01) {
        x = round(sx);
        if (x < 0 || x >= cols) {
            x = 0;
        }
        xdone = true;
    }
    if (sy - round(sy) <= 0.01 || sy - round(sy) <= -0.01) {
        y = round(sy);
        if (y < 0 || y >= rows) {
            y = 0;
        }
        ydone = true;
    }

    // Bilinear interpolation, with intensity 0 for pixels out of image boundary
    if (xdone && ydone) {
        dst[idx] = src[y*cols + x];
    } else if (xdone) {
        int y1 = floor(sy);
        int y2 = floor(sy)+1;
        uint8_t I1, I2;
        if (y1 < 0 || y1 >= rows) {
            I1 = 0;
        } else {
            I1 = src[y1*cols + x];
        }
        if (y2 < 0 || y2 >= rows) {
            I2 = 0;
        } else {
            I2 = src[y2*cols + x];
        }
        dst[idx] = round((y2-sy)*I1 + (sy-y1)*I2);
    } else if (ydone) {
        int x1 = floor(sx);
        int x2 = floor(sx)+1;
        uint8_t I1, I2;
        if (x1 < 0 || x1 >= cols) {
            I1 = 0;
        } else {
            I1 = src[y*cols + x1];
        }
        if (x2 < 0 || x2 >= cols) {
            I2 = 0;
        } else {
            I2 = src[y*cols + x2];
        }
        dst[idx] = round((x2-sx)*I1 + (sx-x1)*I2);
    } else {
        int x1 = floor(sx);
        int x2 = floor(sx)+1;
        int y1 = floor(sy);
        int y2 = floor(sy)+1;
        uint8_t I11, I12, I21, I22;
        if (x1 < 0 || x1 >= cols || y1 < 0 || y1 >= rows) {
            I11 = 0;
        } else {
            I11 = src[y1*cols + x1];
        }
        if (x1 < 0 || x1 >= cols || y2 < 0 || y2 >= rows) {
            I12 = 0;
        } else {
            I12 = src[y2*cols + x1];
        }
        if (x2 < 0 || x2 >= cols || y1 < 0 || y1 >= rows) {
            I21 = 0;
        } else {
            I21 = src[y1*cols + x2];
        }
        if (x2 < 0 || x2 >= cols || y2 < 0 || y2 >= rows) {
            I22 = 0;
        } else {
            I22 = src[y2*cols + x2];
        }
        dst[idx] = round(
            (x2-sx)*(y2-sy)*I11 +
            (x2-sx)*(sy-y1)*I12 +
            (sx-x1)*(y2-sy)*I21 +
            (sx-x1)*(sy-y1)*I22
        );
    }
}

__global__
void disp2Depth(const uint8_t *disp, float *depth,
                    float focalLen, float baselineLen,
                    float minDepth, float maxDepth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (disp[idx] == 0) { // invalidate inf depth
		depth[idx] = 0;
	} else {
        float temp = focalLen * baselineLen / disp[idx];
        if (temp < minDepth || temp > maxDepth) {
            temp = 0; // invalidate depth out of range
        }
        depth[idx] = temp;
	}
}
