from dpcuda import engine
import cv2
import time

left = cv2.imread("test_images/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("test_images/right.png", cv2.IMREAD_GRAYSCALE)

engine.init()
start = time.process_time()
for i in range(100):
    disp = engine.compute(left, right)
print("Runtime for 100 calls of compute_disparity:", time.process_time() - start, "second")
engine.finish()

cv2.imwrite("disp.png", disp)
