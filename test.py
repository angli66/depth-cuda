from dpcuda import DepthEngine
import numpy as np
import cv2
import time

left = cv2.imread("test_images/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("test_images/right.png", cv2.IMREAD_GRAYSCALE)
# scale_percent = 50 # percent of original size
# width = int(left.shape[1] * scale_percent / 100)
# height = int(left.shape[0] * scale_percent / 100)
# dim = (width, height)
# left = cv2.resize(left, dim, interpolation = cv2.INTER_AREA)
# right = cv2.resize(right, dim, interpolation = cv2.INTER_AREA)

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

depthEngine = DepthEngine(img_h, img_w, k_l, k_r, r2l, rectified=True)

start = time.process_time()
for i in range(1): #100
    depth = depthEngine.compute(left, right)
print("Runtime for 100 calls of compute():", time.process_time() - start, "second")

depthEngine.close()

cv2.imwrite("disp.png", depth)
import matplotlib.pyplot as plt
plt.imshow(depth)
plt.show()
