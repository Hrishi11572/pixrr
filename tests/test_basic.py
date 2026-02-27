import pixrr
import numpy as np

def test_gray_conversion():
    img = pixrr.handle_image("test_images/Bikesgray.jpg")
    thrs_img = pixrr.otsu_thresholding(img, inverse=False)
    pixrr.show_image(thrs_img)

test_gray_conversion()