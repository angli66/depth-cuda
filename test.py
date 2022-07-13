from dpcuda import DepthEngine
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

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

start = time.process_time()
for i in range(100):
    depth = depthEngine.compute(left, right)
print("Runtime for 100 calls of compute():", time.process_time() - start, "second")

depthEngine.close()

plt.imshow(depth)
plt.show()