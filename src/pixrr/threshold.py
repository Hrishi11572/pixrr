import numpy as np 
from .io import convert_to_gray
from .utils import img_extremes
from .filters import gaussian_filter, conv2D
import math


def threshold_image(img: np.ndarray | None = None, thresholdValue : int = 0, inverse : bool = False)->np.ndarray:
    '''
    Docstring for threshold_image
    
    :param img: The image as np.ndarray
    :type img: np.ndarray | None
    :param thresholdValue: the pixel intensity value at which you want to threshold the image 
    :type thresholdValue: int
    :param inverse: if you want the blacks to become the whites in a thresholded image
    :type inverse: bool
    :return: a thresholded image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
    if img is None: 
        raise ValueError("Please enter an image and a threshold value")
    
    if not (type(thresholdValue)==int) :
        raise ValueError("threshold value must be an integer")
    
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


def otsu_thresholding(img: np.ndarray, inverse : bool = False)->np.ndarray: 
    '''
    Docstring for otsu_thresholding
    
    :param img: The image as np.ndarray
    :type img: np.ndarray 

    :param inverse: if you want the blacks to become the whites in a thresholded image
    :type inverse: bool
    
    :return: a thresholded image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
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


def iterative_global_thresholding(img: np.ndarray, inverse: bool = False)->np.ndarray: 
    """
    Docstring for iterative_global_thresholding
    
    :param img: the input image 
    :type img: np.ndarray 
    
    :param inverse: True if inversion is required
    :type inverse : bool 
    
    An iterative algorithm to find, optimal threshold value and perform thresholding of gray scale image.
    """
    if img is None: 
        raise ValueError("Please enter an image")
    
    if img.ndim == 3: 
        print("Converted image to gray scale first ... ")
        img = convert_to_gray(img=img)
    
    max_intensity, min_intensity = img_extremes(img)
    
    initial_guess = math.floor(0.5*(max_intensity + min_intensity))
    T = 0
    hist, bin_edges_ = np.histogram(img, bins=256, range=(0,255))
    
    while abs(T - initial_guess) >= 2: 
        m1 = 0
        m2 = 0
        
        group1_size = 0 
        group2_size = 0 
        
        for i in range(1, len(bin_edges_)):
            if i <= initial_guess:
                m1 += bin_edges_[i]*hist[i-1]
                group1_size += hist[i-1]
            else:
                m2 += bin_edges_[i]*hist[i-1]
                group2_size += hist[i-1]
        
        T = initial_guess
        initial_guess = math.floor((m1 + m2)/(group1_size + group2_size))
    
    T = initial_guess
    
    # Now Run Thresholding 
    result = threshold_image(img, thresholdValue=T, inverse=inverse)

    return result


def adaptive_thresh_gaussian(img: np.ndarray, kernel_size :int = 3, a:float = 1.0)->np.ndarray: 
    
    """
    Docstring for adaptive_thresholding 
    """

    if img is None: 
        raise ValueError("Invalid Image passed in input")
    
    if img.ndim == 3: 
        img = convert_to_gray(img)
    
    if kernel_size%2 == 0: 
        raise ValueError("Invalid kernel size, only (odd,odd) shape is preferred.")
    
    kernel = gaussian_filter(kernel_size)
    kernel = kernel / np.sum(kernel)
    blurred = conv2D(
            img,
            mask=kernel,
        )

    threshold = blurred - a
    result = np.where(img > threshold, 255, 0)
    
    return result.astype(np.uint8)
