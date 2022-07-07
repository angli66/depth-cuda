from build import depth_cuda
import cv2
import time

left = cv2.imread("test_images/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("test_images/right.png", cv2.IMREAD_GRAYSCALE)

depth_cuda.init()
start = time.process_time()
for i in range(100):
    disp = depth_cuda.compute_disparity(left, right)
print("Runtime for 100 calls of compute_disparity:", time.process_time() - start, "second")
depth_cuda.finish()

cv2.imwrite("disp.png", disp)
