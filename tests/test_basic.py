import pixrr as pix
import numpy as np
from scipy.spatial import ConvexHull
from scipy import ndimage
import matplotlib.pyplot as plt 

# Obtain an image 

test_image_path = "/Users/hrishikeshtiwari/Desktop/project_root/tests/test_images/sudoku2.png"
img = pix.handle_image(test_image_path)

smoothed_img = pix.gaussian_smoothing(img)
thrs_img = pix.adaptive_thresh_gaussian(smoothed_img, inverse=True)

contour = pix.contour_extractor(thrs_img, save=True, directory="/tests/test_images")


# Code to Obtain individual cells 
height, width = contour.shape[:2]

cell_h = height // 9
cell_w = width // 9

margin_h = int(cell_h * 0.08)
margin_w = int(cell_w * 0.08)

cells = []

for r in range(9):

    row = []

    for c in range(9):

        y1 = r * cell_h
        y2 = (r + 1) * cell_h

        x1 = c * cell_w
        x2 = (c + 1) * cell_w

        cell = img[
            y1 + margin_h : y2 - margin_h,
            x1 + margin_w : x2 - margin_w
        ]

        row.append(cell)

    cells.append(row)


# Convert the cells to gray scale 

cells = np.array(cells)
gray_cells = []

for r in range(cells.shape[0]): 
    for c in range(cells.shape[1]):
        gray_cells.append(pix.convert_to_gray(cells[r][c]))

gray_cells = np.array(gray_cells)

