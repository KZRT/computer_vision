#%%

#%%
import sys
from asyncio import sleep

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read the image using cv2.imread
src = cv2.imread('./assignment-files/distance_transform.jpg')
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, src = cv2.threshold(src, 0, 1, cv2.THRESH_BINARY)


if src is None:
    print('file not found')
    sys.exit(1)

else:
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Euclidean distance transform
dst = cv2.distanceTransform(src, cv2.DIST_L2, 3)
dst = cv2.normalize(dst, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# City block distance transform
dst = cv2.distanceTransform(src, cv2.DIST_L1, 3)
dst = cv2.normalize(dst, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Chessboard distance transform
dst = cv2.distanceTransform(src, cv2.DIST_C, 3)
dst = cv2.normalize(dst, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Euclidean distance transform
def euclidean_distance_transform(src):
    # Make a copy of src
    dst = np.zeros(src.shape, dtype=np.float32)

    # Initialize dst using np.inf to represent infinity
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if src[i, j] == 0:
                dst[i, j] = np.inf

    diagonal_distance = np.sqrt(2)

    # Calculate the distance transform, Top to bottom, left to right
    for i in range(1, dst.shape[0]):
        for j in range(1, dst.shape[1] - 1):
            dst[i, j] = min(dst[i, j],
                            dst[i - 1, j] + 1,
                            dst[i, j - 1] + 1,
                            dst[i - 1, j - 1] + diagonal_distance,
                            dst[i - 1, j + 1] + diagonal_distance)

    # Calculate the distance transform, Bottom to top, right to left
    for i in range(dst.shape[0] - 2, -1, -1):
        for j in range(dst.shape[1] - 2, -1, -1):
            dst[i, j] = min(dst[i, j],
                            dst[i + 1, j] + 1,
                            dst[i, j + 1] + 1,
                            dst[i + 1, j + 1] + diagonal_distance,
                            dst[i + 1, j - 1] + diagonal_distance)

    # Normalize the distance transform
    max_value = np.nanmax(dst[dst != np.inf])
    dst = dst / max_value

    dst[dst == np.inf] = 1
    dst[dst == np.nan] = 0
    dst = np.uint8(dst * 255)
    dst = 255 - dst

    return dst


# City Block distance transform
def city_block_distance_transform(src):
    # Make a copy of src
    dst = np.zeros(src.shape, dtype=np.float32)

    # Initialize dst using np.inf to represent infinity
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if src[i, j] == 0:
                dst[i, j] = np.inf

    diagonal_distance = 2

    # Calculate the distance transform, Top to bottom, left to right
    # from i, j , AL is (i - 1 , j), (i, j - 1), (i - 1, j - 1), (i - 1, j + 1)
    for i in range(1, dst.shape[0]):
        for j in range(1, dst.shape[1] - 1):
            dst[i, j] = min(dst[i, j],
                            dst[i - 1, j] + 1,
                            dst[i, j - 1] + 1,
                            dst[i - 1, j - 1] + diagonal_distance,
                            dst[i - 1, j + 1] + diagonal_distance)

    # Calculate the distance transform, Bottom to top, right to left
    # from i, j , BR is (i + 1 , j), (i, j + 1), (i + 1, j + 1), (i + 1, j - 1)
    for i in range(dst.shape[0] - 2, -1, -1):
        for j in range(dst.shape[1] - 2, -1, -1):
            dst[i, j] = min(dst[i, j],
                            dst[i + 1, j] + 1,
                            dst[i, j + 1] + 1,
                            dst[i + 1, j + 1] + diagonal_distance,
                            dst[i + 1, j - 1] + diagonal_distance)

    # Normalize the distance transform
    max_value = np.nanmax(dst[dst != np.inf])
    dst = dst / max_value

    dst[dst == np.inf] = 1
    dst[dst == np.nan] = 0
    dst = np.uint8(dst * 255)
    dst = 255 - dst

    return dst


# Chessboard distance transform
def chessboard_distance_transform(src):
    # Make a copy of src
    dst = np.zeros(src.shape, dtype=np.float32)

    # Initialize dst using np.inf to represent infinity
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if src[i, j] == 0:
                dst[i, j] = np.inf

    diagonal_distance = 1

    # Calculate the distance transform, Top to bottom, left to right
    for i in range(1, dst.shape[0]):
        for j in range(1, dst.shape[1] - 1):
            dst[i, j] = min(dst[i, j],
                            dst[i - 1, j] + 1,
                            dst[i, j - 1] + 1,
                            dst[i - 1, j - 1] + diagonal_distance,
                            dst[i - 1, j + 1] + diagonal_distance)

    # Calculate the distance transform, Bottom to top, right to left
    for i in range(dst.shape[0] - 2, -1, -1):
        for j in range(dst.shape[1] - 2, -1, -1):
            dst[i, j] = min(dst[i, j],
                            dst[i + 1, j] + 1,
                            dst[i, j + 1] + 1,
                            dst[i + 1, j + 1] + diagonal_distance,
                            dst[i + 1, j - 1] + diagonal_distance)

    # Normalize the distance transform
    max_value = np.nanmax(dst[dst != np.inf])
    dst = dst / max_value

    dst[dst == np.inf] = 1
    dst[dst == np.nan] = 0
    dst = np.uint8(dst * 255)
    dst = 255 - dst

    return dst


dst = euclidean_distance_transform(src)
dst = cv2.equalizeHist(dst)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

dst = city_block_distance_transform(src)
dst = cv2.equalizeHist(dst)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

dst = chessboard_distance_transform(src)
dst = cv2.equalizeHist(dst)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

#%%
