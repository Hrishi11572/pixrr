import pixrr as pix
import numpy as np


# obtain an image 

test_image_path = "/Users/hrishikeshtiwari/Desktop/project_root/tests/test_images/testfile2.jpg"
img = pix.handle_image(test_image_path)

# obtain imge summary 

pix.imageSummary(img)