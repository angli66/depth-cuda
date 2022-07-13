from dpcuda import DepthEngine
import numpy as np
import cv2

left = cv2.imread("test_images/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("test_images/right.png", cv2.IMREAD_GRAYSCALE)

scale_percent = 50 # percent of original size
width = int(left.shape[1] * scale_percent / 100)
height = int(left.shape[0] * scale_percent / 100)
dim = (width, height)
left_half = cv2.resize(left, dim, interpolation = cv2.INTER_AREA)
right_half = cv2.resize(right, dim, interpolation = cv2.INTER_AREA)

k_l = np.array([
    [920., 0., 640.],
    [0., 920., 360.],
    [0., 0., 1.]
])
k_r = k_l
l2r = np.array([
    [1., 0, 0, 0.0545],
    [0, 1., 0, 0],
    [0, 0, 1., 0],
    [0, 0, 0, 1.]
])
r2l = np.linalg.inv(l2r)

for i in range(1000):
    if i % 2 == 0:
        depthEngine = DepthEngine(*left.shape, k_l, k_r, r2l, min_depth=0.2, max_depth=10.0, rectified=False)
        depth = depthEngine.compute(left, right)
        depthEngine.close()
    else:
        depthEngine = DepthEngine(*left_half.shape, k_l, k_r, r2l, min_depth=0.2, max_depth=10.0, rectified=False)
        depth = depthEngine.compute(left_half, right_half)
        depthEngine.close()

print("pass!")
