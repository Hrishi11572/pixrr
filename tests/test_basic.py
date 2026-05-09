import pixrr as pix
import numpy as np


# obtain an image 

test_image_path = "/Users/hrishikeshtiwari/Desktop/project_root/tests/test_images/sudoku2.png"
img = pix.handle_image(test_image_path)

thrs_img = pix.adaptive_thresh_gaussian(img)
smoothed_img = pix.gaussian_smoothing(thrs_img)

