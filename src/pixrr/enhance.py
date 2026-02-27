import numpy as np 
import matplotlib.pyplot as plt 
from .io import convert_to_gray

def linear_contrast_enhancement(img : np.ndarray, low: int = 0, high : int = 255 , save : bool = False, filename : str | None = None)->np.ndarray:
    
    if img.ndim == 3: 
         # convert colored to gray scale 
         img = convert_to_gray(img)
    
    if max(low, high) > 255 or min(low, high) < 0: 
        raise ValueError("{low} and {high} should be between 0 and 255")
    
    new_img = np.asarray(img)
    
    min_intensity = img.min()
    max_intensity = img.max()
    
    # Edge case (Zero denominator)
    if max_intensity == min_intensity:
        return np.full_like(img, low, dtype=np.uint8)

    new_img = ((high - low)/(max_intensity - min_intensity)) * (img - min_intensity) + low
    new_img = new_img.astype(np.uint8)
    
    fig, ax = plt.subplot_mosaic([
        ['original', 'enhanced']
    ], figsize=(7, 3.5))

    ax["original"].imshow(img, cmap="gray")
    ax["original"].axis("off")
    ax["original"].set_title("Original Image")
    
    

    ax["enhanced"].imshow(new_img, cmap="gray")
    ax["enhanced"].axis("off")
    ax["enhanced"].set_title("Enhanced Image")
    plt.show()
    
    if save: 
        if filename is None: 
            raise ValueError("IF you want to save the image, please pass a filename")
        else: 
            # 1. Create a new figure and axes for saving
            fig2, ax2 = plt.subplots()
            
            # 2. Draw the new image onto the *saving* axes
            ax2.imshow(new_img, cmap="gray")
            
            # 3. Configure the axes
            ax2.axis("off")
            
            HIGH_RES_DPI = 600
            
            # **The key line for high resolution is here:**
            fig2.savefig(
                filename, 
                dpi=HIGH_RES_DPI,              # Sets the resolution
                bbox_inches="tight",           # Crops unnecessary white space
                pad_inches=0                   # Removes padding
            )            
            plt.close(fig2)
    return new_img



def histogram_equalization(img : np.ndarray, low: int = 0, high : int  = 255, save : bool = False, filename : str | None = None): 
     
    # convert to gray scale if necessary 
    if img.ndim == 3: 
        img = convert_to_gray(img)
    
    if not (0 <= low < high <= 255):
        raise ValueError(f"{low} and {high} must satisfy 0 <= low < high <= 255")

    hist , _ = np.histogram(img, bins=256, range=(0,256))
    hist = hist/hist.sum()
    
    cdf_original = np.cumsum(hist)
    
    new_img = np.round(cdf_original[img] * (high - low) + low).astype(np.uint8)

    
    # display the enhanced image 
    fig, ax = plt.subplot_mosaic([
        ['original', 'enhanced']
    ], figsize=(7, 3.5))

    ax["original"].imshow(img, cmap="gray")
    ax["original"].axis("off")
    ax["original"].set_title("Original Image")
    
    
    ax["enhanced"].imshow(new_img, cmap="gray")
    ax["enhanced"].axis("off")
    ax["enhanced"].set_title("Enhanced Image")
    plt.show()
    plt.close(fig)

    # save the new image if asked to do so 
    if save: 
        if filename is None: 
            raise ValueError("IF you want to save the image, please pass a filename")
        else: 
            # 1. Create a new figure and axes for saving
            fig2, ax2 = plt.subplots()
            
            # 2. Draw the new image onto the *saving* axes
            ax2.imshow(new_img, cmap="gray")
            
            # 3. Configure the axes
            ax2.axis("off")
            
            HIGH_RES_DPI = 600
            
            # **The key line for high resolution is here:**
            fig2.savefig(
                filename, 
                dpi=HIGH_RES_DPI,              # Sets the resolution
                bbox_inches="tight",           # Crops unnecessary white space
                pad_inches=0                   # Removes padding
            )            
            plt.close(fig2)
    
    return new_img