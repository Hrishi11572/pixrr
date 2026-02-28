import pixrr
import numpy as np


# obtain an image 

test_image_path = "/Users/hrishikeshtiwari/Desktop/project_root/tests/test_images/testfile.png"
img = pixrr.handle_image(test_image_path)


# see the histogram 

pixrr.plot_img_hist(img, channel="green", save=True, directory="/Users/hrishikeshtiwari/Desktop/project_root/tests/test_images/", filename="georgebush.png")

# # get the coordinates to test crop and save image functions 
# coords = (0,0,1000,5000)
# cropped_img = pixrr.crop_image(img, coords , viewMode=False)
# pixrr.show_image(cropped_img)