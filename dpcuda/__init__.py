import dpcuda.engine
import numpy as np
import cv2

def init(h, w, kl, kr, rt, distortl=None, distortr=None):
    """
    :param h: Image height
    :param w: Image width
    :param kl: Left intrinsic matrix
    :param kr: Right intrinsic matrix
    :param rt: Extrinsic matrix (left to right)
    :param distortr: Left distortion coefficients
    :param distortl: Right distortion coefficients
    """
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        R=rt[:3, :3], T=rt[:3, 3:],
        cameraMatrix1=kl, cameraMatrix2=kr,
        alpha=1.0, imageSize=(w, h), newImageSize=(w, h),
        distCoeffs1=distortl, distCoeffs2=distortr
    )

    map1 = cv2.initUndistortRectifyMap(kl, distortr, r1, p1, (w, h), cv2.CV_32F)
    map2 = cv2.initUndistortRectifyMap(kr, distortl, r2, p2, (w, h), cv2.CV_32F)

    dpcuda.engine.init()

def compute():
    pass