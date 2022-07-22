 # By Ang Li, Jul 2022

import numpy as np
import cv2
import dpcuda.engine

# NOTICE: this class is a wrapper for underlying cuda code. No matter how many instances are created, they
# will point to the same cuda instance in memory
class DepthEngine:
    def __init__(self, img_h, img_w, k_l, k_r, r2l, min_depth, max_depth,
                    dist_l=None, dist_r=None, rectified=False,
                    p1_penalty=10, p2_penalty=120, census_width=9, census_height=7):
        """
        :param img_h: Image height, must be divisible by 4 and greater than or equal to 32
        :param img_w: Image width, must be divisible by 4 and greater than or equal to 32
        :param k_l: Left intrinsic matrix
        :param k_r: Right intrinsic matrix
        :param r2l: Extrinsic matrix (right to left)
        :param min_depth: minimum valid depth
        :param max_depth: maximum valid depth
        :param dist_l: Left distortion coefficients
        :param dist_r: Right distortion coefficients
        :param rectified: Whether the input has already been rectified
        :param p1_penalty: p1 penalty for semi-global matching, must be integer less than p2_penalty
        :param p2_penalty: p2 penalty for semi-global matching, must be integer less than 224
        :param census_width: width of the census transform window, census_width*census_height <= 65
        :param census_height: height of the census transform window, census_height*census_height <= 65
        """
        if not isinstance(img_h, int) or not isinstance(img_w, int) or \
                img_h % 4 != 0 or img_w % 4 != 0 or img_h < 32 or img_w < 32:
            raise TypeError("Image height and width must be integer divisible by 4 and no less than 32")

        if not isinstance(p1_penalty, int) or not isinstance(p2_penalty, int) or \
                p1_penalty >= p2_penalty or p2_penalty >= 224:
            raise TypeError("p1 must be integer less than p2 and p2 be integer less than 224")

        if not isinstance(census_width, int) or not isinstance(census_height, int) or \
                census_width % 2 == 0 or census_height % 2 == 0 or census_width*census_height > 65:
            raise TypeError("Census width/height must be odd integers and their product should be no larger than 65")

        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        R=r2l[:3, :3], T=r2l[:3, 3:],
        cameraMatrix1=k_l, cameraMatrix2=k_r,
        alpha=1.0, imageSize=(img_w, img_h), newImageSize=(img_w, img_h),
        distCoeffs1=dist_l, distCoeffs2=dist_r
        )
        f_len = np.abs(q[2][3]) # focal length of the left camera (in meters)
        b_len = 1.0 / np.abs(q[3][2]) # baseline length
        map_l = cv2.initUndistortRectifyMap(k_l, dist_l, r1, p1, (img_w, img_h), cv2.CV_32F)
        map_r = cv2.initUndistortRectifyMap(k_r, dist_r, r2, p2, (img_w, img_h), cv2.CV_32F)
        map_lx, map_ly = map_l
        map_rx, map_ry = map_r

        dpcuda.engine.init(img_h, img_w, f_len, b_len, min_depth, max_depth,
                            map_lx, map_ly, map_rx, map_ry, rectified,
                            p1_penalty, p2_penalty, census_width, census_height)

    def compute(self, img_l, img_r):
        """
        :param img_l: Left infrared/grayscale image (uint8) captured by depth sensor
        :param img_r: Right infrared/grayscale image (uint8) captured by depth sensor
        :return depth: Calculated depth map
        """
        depth = dpcuda.engine.compute(img_l, img_r)

        return depth
    
    def close(self):
        """
        Free resources including memory allocated by cuda.
        """
        dpcuda.engine.close()
