import pixrr
import numpy as np


# obtain an image 

test_image_path = "/Users/hrishikeshtiwari/Desktop/project_root/tests/test_images/Bikesgray.jpg"
img = pixrr.handle_image(test_image_path)


# do some thresholding and contour extraction 

img2 = pixrr.threshold_image(img,50,inverse=True)
pixrr.contour_extractor(img2,save=True)


# # get the coordinates to test crop and save image functions 
# coords = (0,0,1000,5000)
# cropped_img = pixrr.crop_image(img, coords , viewMode=False)
# pixrr.show_image(cropped_img)