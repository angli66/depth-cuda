from dpcuda import engine
import cv2
import time

left = cv2.imread("test_images/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("test_images/right.png", cv2.IMREAD_GRAYSCALE)

engine.init(*left.shape) # there is overhead when first time starting cuda
engine.finish()

start = time.process_time()
engine.init(*left.shape)
print("Runtime for initiate:", time.process_time() - start, "second")

scale_percent = 50 # percent of original size
width = int(left.shape[1] * scale_percent / 100)
height = int(left.shape[0] * scale_percent / 100)
dim = (width, height)

start = time.process_time()
for i in range(100):
    # if i%2 == 0:
    #     # resize image
    #     l = cv2.resize(left, dim, interpolation = cv2.INTER_AREA)
    #     r = cv2.resize(right, dim, interpolation = cv2.INTER_AREA)
    #     engine.init(*l.shape)
    #     disp = engine.compute(l, r)

    engine.init(*left.shape)
    disp = engine.compute(left, right)
print("Runtime for 100 calls of compute_disparity:", time.process_time() - start, "second")
engine.finish()

cv2.imwrite("disp.png", disp)
