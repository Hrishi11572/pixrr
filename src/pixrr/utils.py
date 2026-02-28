# This file deals with any additional utility needed ex : image-cropping and image-rotation and color-space interconversion 


import numpy as np 
import os 
from PIL import Image
from .io import show_image, save_image

def crop_image(img : np.ndarray, coords : tuple , directory : str, filename: str , viewMode : True)->np.ndarray: 
    """
    coords : (startx , starty , endx , endy)
    """
    if img is None: 
        raise ValueError("No image passed. Please ensure proper input")
    
    cropped_img = img[coords[1] : coords[3] , coords[0]: coords[2]]
    
    if cropped_img.size == 0: 
        print("Error: Cropping coordinates resulted in an empty image.")
        return

    # save the image 
    save_image(cropped_img, directory=directory, filename=filename)
    
    # display the image 
    if viewMode: 
        show_image(cropped_img)    
    
    return cropped_img
