import numpy as np 
from .io import convert_to_gray
import random


def threshold_image(img: np.ndarray | None = None, thresholdValue : int = 0, inverse : bool = False)->np.ndarray:
    '''
    Docstring for threshold_image
    
    :param img: The image as np.ndarray
    :type img: np.ndarray | None
    :param thresholdValue: the pixel value at which you want to threshold the image 
    :type thresholdValue: int
    :param inverse: if you want the blacks to become the whites in a thresholded image
    :type inverse: bool
    :return: a thresholded image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
    if img is None: 
        raise ValueError("Please enter an image and a threshold value")
    
    if not (0 <= thresholdValue <= 255):
        raise ValueError("thresholdValue must be between 0 and 255.")
    
    if img.ndim == 2: 
        gray_img = img 
    else:
        # convert to grayscale first 
        try:
            gray_img = convert_to_gray(img=img)
        except Exception as e: 
            raise IOError(f"Can't convert to grayscale : {e}")
    
    if not inverse:
        result = np.where(gray_img < thresholdValue, 0, 255)
    else: 
        result = np.where(gray_img < thresholdValue, 255, 0)
    
    return result.astype(np.uint8)


def otsu_thresholding(img: np.ndarray, inverse : bool = False): 
    def otsu_intraclass_variance(img: np.ndarray , threshold: int):
        ''' https://en.wikipedia.org/wiki/Otsu%27s_method ''' 
        return np.nansum(
            [
                np.mean(cls) * np.var(img, where=cls)
                #   weight   ·  intra-class variance
                for cls in [img >= threshold, img < threshold]
            ]
        )
    
    otsu_threshold = min(
        range(np.min(img) + 1, np.max(img)),
        key=lambda th: otsu_intraclass_variance(img, th),
    )
    
    return threshold_image(img=img, thresholdValue=otsu_threshold, inverse=inverse)