import numpy as np
import matplotlib.pyplot as plt

result = plt.imread("result.png")
example_result = plt.imread("test_images/example_result.png")

if np.array_equal(result[:, 200:], example_result[:, 200:]):
    print("pass!")