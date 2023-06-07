import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
src = cv2.imread('./assignment-files/under_lena.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('file not found')
    sys.exit(1)
else:
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Make histogram
hist = cv2.calcHist([src], [0], None, [256], [0, 256])

# Plot histogram
plt.plot(hist)
plt.show()

# Make histogram function without using cv2.calcHist
def calculate_histogram(src):
    # 256X256 image to histogram shrinks to 1D array of 256
    result = np.zeros(256, dtype=np.uint32)

    # Iterate over the image, use Hi[img[i][j]]++
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # Python does not support ++ operator
            result[src[i, j]] += 1
    return result


my_hist = calculate_histogram(src)

# Plot histogram
plt.plot(my_hist)
plt.show()

# Equalize histogram
dst = cv2.equalizeHist(src)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Equalize histogram
def equalize_histogram(src, hist):
    # Calculate cumulative histogram
    cdf = np.zeros(256, dtype=np.uint32)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Calculate LUT
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.uint8(255 * cdf[i] / cdf[255])

    # Apply LUT
    dst = np.zeros(src.shape, dtype=np.uint8)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = lut[src[i, j]]
    return dst


my_dst = equalize_histogram(src, my_hist)
cv2.imshow('my_dst', my_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
