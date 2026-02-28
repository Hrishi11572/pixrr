import pixrr
import numpy as np


# obtain an image 

test_image_path = "tests/test_images/Bikesgray.jpg"
img = pixrr.handle_image(test_image_path)

# get the coordinates to test crop and save image functions 

coords = (0,0,500,500)

cropped_img = crop_image(img, coords , "/tests/test_images", "bullshit.png", viewMode=True)