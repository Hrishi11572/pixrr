import numpy as np 
from numba import njit 


''' Defining Convolution Here '''

@njit
def padd_image(img: np.ndarray, row_pad: int = 1, col_pad: int = 1)->np.ndarray:
    '''
    docstring for padd_image: 
    
    :param img : input image to be padded as a numpy array 
    :type img : np.ndarray 
    :param row_pad: the number of rows to padd (default is 1)
    :type row_pad: int 
    :param col_pad: the number of columns to padd (default is 1)
    :type col_pad: int 
    :return: padded image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
    h, w = img.shape
    ph, pw = h + 2*row_pad, w + 2*col_pad
    padded = np.zeros((ph, pw), dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            padded[i + row_pad, j + col_pad] = img[i, j]
    return padded


@njit
def conv2D(img: np.ndarray, mask : np.ndarray, hstep: int = 1, vstep: int = 1)->np.ndarray: 
    '''
    docstring for conv2D: 
    
    :param img : input image to be convolved by a mask as a numpy array 
    :type img : np.ndarray 
    :param mask: the kernel which would be convolved onto the image 
    :type mask: np.ndarray 
    :param hstep: horisontal stride 
    :type hstep: int 
    :param vstep: vertical stride 
    :type vstep: int  
    :return: returns convolved image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
    
    if not (mask.shape[0] % 2 == 1 and mask.shape[1]%2 == 1): 
        raise ValueError("mask should have sape of the form : (odd_val, odd_val)")
    
    padded_img = padd_image(img, mask.shape[0]//2 , mask.shape[1]//2)
    
    
    if img.ndim == 2: 
        kernel_height, kernel_width = mask.shape
        padded_height, padded_width = padded_img.shape 
        
        output_height = (padded_height - kernel_height) // vstep + 1
        output_width = (padded_width - kernel_width) // hstep + 1
        
        new_img = np.zeros((output_height, output_width), dtype="float64")
        
        for y in range(0, output_height): 
            for x in range(0, output_width): 
                region = padded_img[y*vstep: y*vstep + kernel_height, x*hstep : x*hstep + kernel_width]
                new_img[y][x] = np.sum(region * mask)
        
        return new_img
    
    elif img.ndim == 3: 
        channels = []
        
        for i in range(3):
            channel = conv2D(img[:,:,i], mask, hstep, vstep)
            channels.append(channel)
        
        return np.stack(channels, axis=2)
    else: 
        raise ValueError("Unsupported image type!")
    
    
@njit
def laplacian(img : np.ndarray)->np.ndarray:
    '''
    docstring for laplacian 
    
    :param img: input image, which we are going to convolve with laplacian kernel 
    :type img: np.ndarray 
    :return: returns the convolved image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
    kernel_1 = np.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1 , 0]
            ]
        ).astype(np.float32)
        
    kernel_2 =  np.array(
            [
                [1, 0, 1],
                [0, -4, 0],
                [1, 0 , 1]
            ]
        ).astype(np.float32)
        
    kernel = (kernel_1 + kernel_2 * 4)/5
    
    if img.ndim == 2: 
        return conv2D(img=img, mask=kernel).astype(np.float32)
    
    elif img.ndim == 3: 

        new_img = np.zeros(img.shape, dtype=np.float32)     
        
        for c in range(3):
            new_img[:,:,c] = conv2D(img[:,:,c], mask=kernel)
            
        return new_img


def sharpen_image(img: np.ndarray, c : float = 1.0)->np.ndarray: 
    '''
    docstring for sharpen_image 
    
    :param img : input image 
    :type img : np.ndarray 
    :param c : the control parameter for sharpening 
    :type c : float (default is 1.0)
    :return: returns the sharpened image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    
    Additional description 
    sharpened_Image = I - c·∇²I
    '''
    img_F = img.astype(np.float32)
    lap = laplacian(img)
    
    sharpImage = img_F - c * lap 
    sharpImage = np.clip(sharpImage, 0, 255)
    
    return sharpImage.astype(np.uint8)
    # return (img - c * laplacian(img)).astype(np.uint8) -- without clipping is bad



@njit
def gaussian_filter(sigma: float = 1.0, size: int = 3)->np.ndarray: 
    '''
    docstring for gaussian_filter
    
    :param sigma : the parameter for gaussian kernel   
    :type sigma : float 
    :param size: the size of the kernel 
    :type size : int (default is 3, only odd values are permissible)
    :return: returns the size * size kernel 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    
    For additional details kindly access 
        https://stackoverflow.com/a/43346070
    '''
    
    if size % 2 == 0: 
        raise ValueError("Kernel sizes should be odd")
    
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def gaussian_smoothing(img : np.ndarray, kernel_size: int = 3, sigma : float = 1.0)->np.ndarray:
    '''
    docstring for gaussian_smoothing
    
    :param img : the input image as numpy array 
    :type img : np.ndarray 
    :param kernel_size: the size of the kernel 
    :type kernel_size : int (default is 3, only odd values are permissible)
    :param sigma : the parameter of the kernel 
    :type sigma : float (default is 1.0)
    :return: returns the smoothed image
    :rtype: ndarray[_AnyShape, dtype[Any]]
    
    '''
    gauss = gaussian_filter(sigma, kernel_size)
    
    smoothed = conv2D(img, gauss)
    smoothed = np.clip(smoothed, 0, 255)
    
    return smoothed.astype(np.uint8)
