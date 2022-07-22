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

depthEngine = DepthEngine(img_h, img_w, k_l, k_r, r2l, 0.2, 10.0,
                            p1_penalty=10, p2_penalty=120,
                            census_width=9, census_height=7)

correct = True
for i in range(100):
    depth1 = depthEngine.compute(left, right)
    depth2 = depthEngine.compute(left, right)
    if np.array_equal(depth1, depth2):
        continue
    else:
        correct = False
        print("fail!")
        plt.imsave("result1.png", depth1)
        plt.imsave("result2.png", depth2)
        break
    
if correct:
    print("pass!")

depthEngine.close()
