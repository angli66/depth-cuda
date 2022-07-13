from dpcuda import DepthEngine
import numpy as np
import matplotlib.pyplot as plt
import cv2

left = cv2.imread("test_images/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("test_images/right.png", cv2.IMREAD_GRAYSCALE)

img_h, img_w = left.shape[0], left.shape[1]
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

depthEngine = DepthEngine(img_h, img_w, k_l, k_r, r2l, min_depth=0.2, max_depth=10.0, rectified=False)

# for i in range(10):
#     depth = depthEngine.compute(left, right)
#     depth = depth[:, 200:]
#     if i == 0:
#         prev = depth
#     else:
#         print(np.array_equal(prev, depth))
#         prev = depth

depth1 = depthEngine.compute(left, right)
depth1 = depth1[:, 200:]
depth2 = depthEngine.compute(left, right)
depth2 = depth2[:, 200:]
depth3 = depthEngine.compute(left, right)
depth3 = depth3[:, 200:]
print(np.array_equal(depth1, depth2))

plt.imsave("depth1.png", depth1)
plt.imsave("depth2.png", depth2)

depthEngine.close()
