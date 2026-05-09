# This file deals with any additional utility needed ex : image-cropping and image-rotation and color-space interconversion 


import numpy as np 
import os 
from PIL import Image
from .io import show_image, save_image

def crop_image(img : np.ndarray, coords : tuple , viewMode : bool = True)->np.ndarray: 
    '''
    Docstring for crop_image
    
    :param img: the input image 
    :type img: np.ndarray

    :para coords : the coordinates required for cropping in the form (start_x, start_y, end_x , end_y)
    :type coords : tuple 
    
    - crops the image as per the provided coordinates 
    '''
    
    if img is None: 
        raise ValueError("No image passed. Please ensure proper input")
    
    cropped_img = img[coords[1] : coords[3] , coords[0]: coords[2]]
    
    if cropped_img.size == 0: 
        print("Error: Cropping coordinates resulted in an empty image.")
        return
    
    # display the image 
    if viewMode: 
        show_image(cropped_img)    
                
    return cropped_img


def imgExtremes(img : np.ndarray)->tuple: 
    ''' 
    Docstring for imgExtremes: 
    
    :param img: the input image 
    :type img: np.ndarray 
    
    Returns a tuple, containing the maximum and the minimum pixel intensity  
    '''
    if img is None: 
        raise ValueError("Please enter a valid image")
    
    return (np.max(img), np.min(img))

def imageSummary(img:np.ndarray)->None: 
    '''
        Docstring for imageSummary
        
        :param img: the input image 
        :type img : np.ndarray 
        
        Provides the summary of the image, details about the dimension, the channels, max_intensity, min_intensity
    '''
    
    print("\n##########################################")
    print("################ SUMMARY #################")
    print("##########################################\n")
    
    print("- Colored image") if img.ndim == 3 else print("- Gray Scale Image")

    max_i, min_i = imgExtremes(img)
    print(f"- Max Pixel intensity : {max_i}, Min Pixel intensity : {min_i}")        
    print(f"- Image shape : {img.shape}")
    return None