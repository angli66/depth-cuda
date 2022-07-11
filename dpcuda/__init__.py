 # By Ang Li, Jul 2022

import dpcuda.engine
import numpy as np
import cv2

class DepthEngine:
    def __init__(self, img_h, img_w, k_l, k_r, r2l, min_depth, max_depth, dist_l=None, dist_r=None, rectified=False):
        """
        :param img_h: Image height
        :param img_w: Image width
        :param k_l: Left intrinsic matrix
        :param k_r: Right intrinsic matrix
        :param r2l: Extrinsic matrix (right to left)
        :param min_depth: minimum valid depth
        :param max_depth: maximum valid depth
        :param dist_l: Left distortion coefficients
        :param dist_r: Right distortion coefficients
        :param rectified: Whether the input has already been rectified
        """
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        R=r2l[:3, :3], T=r2l[:3, 3:],
        cameraMatrix1=k_l, cameraMatrix2=k_r,
        alpha=1.0, imageSize=(img_w, img_h), newImageSize=(img_w, img_h),
        distCoeffs1=dist_l, distCoeffs2=dist_r
        )

        self._rectified = rectified # whether the input images are already rectified
        self._q = q
        self._map1 = cv2.initUndistortRectifyMap(k_l, dist_l, r1, p1, (img_w, img_h), cv2.CV_32F)
        self._map2 = cv2.initUndistortRectifyMap(k_r, dist_r, r2, p2, (img_w, img_h), cv2.CV_32F)

        f_len = np.abs(q[2][3]) # focal length of the left camera (in meters)
        b_len = 1.0 / np.abs(q[3][2]) # baseline length
        dpcuda.engine.init(img_h, img_w, f_len, b_len, min_depth, max_depth)

    def compute(self, img_l, img_r):
        """
        :param img_l: Left infrared/grayscale image (uint8) captured by depth sensor
        :param img_r: Right infrared/grayscale image (uint8) captured by depth sensor
        :return depth: Calculated depth map
        """
        if not self._rectified:
            img_l = cv2.remap(img_l, *self._map1, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, *self._map2, cv2.INTER_LINEAR)

        depth = dpcuda.engine.compute(img_l, img_r)

        return depth
    
    def close(self):
        """
        Free resources including memory allocated by cuda.
        """
        dpcuda.engine.close()
