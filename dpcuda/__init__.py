import dpcuda.engine
import numpy as np
import cv2

class DepthEngine:
    def __init__(self, img_h, img_w, k_l, k_r, r2l, dist_l=None, dist_r=None, rectified=False):
        """
        :param img_h: Image height
        :param img_w: Image width
        :param k_l: Left intrinsic matrix
        :param k_r: Right intrinsic matrix
        :param r2l: Extrinsic matrix (right to left)
        :param dist_l: Left distortion coefficients
        :param dist_r: Right distortion coefficients
        """
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        R=r2l[:3, :3], T=r2l[:3, 3:],
        cameraMatrix1=k_l, cameraMatrix2=k_r,
        alpha=1.0, imageSize=(img_w, img_h), newImageSize=(img_w, img_h),
        distCoeffs1=dist_l, distCoeffs2=dist_r
        )

        self._rectified = rectified
        self._q = q
        self._map1 = cv2.initUndistortRectifyMap(k_l, dist_l, r1, p1, (img_w, img_h), cv2.CV_32F)
        self._map2 = cv2.initUndistortRectifyMap(k_r, dist_r, r2, p2, (img_w, img_h), cv2.CV_32F)

        dpcuda.engine.init(img_h, img_w)

    def compute(self, img_l, img_r):
        """
        :param img_l: Left infrared/grayscale image captured by depth sensor
        :param img_r: Right infrared/grayscale image captured by depth sensor
        :return depth: Calculated depth map
        """
        if not self._rectified:
            img_l = cv2.remap(img_l, *self._map1, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, *self._map2, cv2.INTER_LINEAR)

        disp = dpcuda.engine.compute(img_l, img_r)

        # disp to depth test code
        mask = disp >= 1
        _3d_image = cv2.reprojectImageTo3D(disp, self._q)
        depth = _3d_image[..., 2]
        depth = _3d_image[..., 2]
        depth[~mask] = 0
        depth[np.isinf(depth)] = 0
        depth[np.isnan(depth)] = 0
        max_depth = 10.0
        min_depth = 0.2
        depth[depth > max_depth] = 0
        depth[depth < min_depth] = 0
        _3d_image[..., 2] = depth

        # import open3d as o3d
        # points = _3d_image.reshape(-1, 3)
        # valid_flag = mask.reshape(-1)
        # valid_points = []
        # for i, point in enumerate(points):
        #     if valid_flag[i]:
        #         valid_points.append(point)
        # valid_points = o3d.utility.Vector3dVector(np.array(valid_points))
        # pointcloud = o3d.geometry.PointCloud(points=valid_points)
        # o3d.visualization.draw_geometries([pointcloud])

        return depth
    
    def close(self):
        """
        Free resources including memory allocated by cuda.
        """
        dpcuda.engine.close()
