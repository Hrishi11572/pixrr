import numpy as np 
import matplotlib.pyplot as plt 
from .io import convert_to_gray
from .filters import conv2D

def gradient_prewitt(img: np.ndarray,kernel_size : int = 3, direction: str = "both", hstep: int = 1, vstep :int = 1) -> np.ndarray : 
    if img.ndim == 3: 
        img = convert_to_gray(img)
    
    if kernel_size % 2 == 0: 
        raise ValueError("Mask should be of the following form (odd, odd)")
    
    # create the prewitt kernel of the given size 
    prewitt_horizontal = np.zeros((kernel_size, kernel_size-2))
    prewitt_horizontal = np.hstack((prewitt_horizontal, np.full((kernel_size,1), -1)))
    prewitt_horizontal = np.hstack((prewitt_horizontal[:,::-1], np.full((kernel_size,1), fill_value=1))) 

    prewitt_vertical = prewitt_horizontal.T[::-1,:] 
    
    if direction == "h": 
        horizontal_gradient = conv2D(img, prewitt_horizontal, hstep , vstep)
        return horizontal_gradient
    
    elif direction == "v": 
        vertical_gradient = conv2D(img, prewitt_vertical, hstep, vstep)
        return vertical_gradient
    
    elif direction == "both": 
        horizontal_gradient = conv2D(img, prewitt_horizontal, hstep, vstep)
        vertical_gradient = conv2D(img, prewitt_vertical, hstep, vstep)
        grad= np.sqrt(horizontal_gradient **2 + vertical_gradient **2)
        grad = (grad / grad.max() * 255)
        return grad.astype(np.uint8)
    


def gradient_sobel(img: np.ndarray,kernel_size : int = 3, direction: str = "both", hstep: int = 1, vstep :int = 1) -> np.ndarray : 
    if img.ndim == 3: 
        img = convert_to_gray(img)
    
    if kernel_size % 2 == 0: 
        raise ValueError("Mask should be of the following form (odd, odd)")
    
    # create the prewitt kernel of the given size 
    def get_sobel_kernels(size: int):
        if size % 2 == 0 or size < 3:
            raise ValueError("Size must be odd and at least 3")

        # 1. Generate Pascal's Triangle row for smoothing
        def get_pascal_row(n):
            row = [1]
            for k in range(n):
                row.append(row[k] * (n - k) // (k + 1))
            return np.array(row, dtype=np.float32)

        # Smoothing vector (s)
        s = get_pascal_row(size - 1)
        
        # Derivative vector (d) 
        # Logic: difference of the Pascal row one degree smaller
        d_prev = get_pascal_row(size - 2)
        d = np.zeros(size)
        d[:-1] += d_prev
        d[1:] -= d_prev

        # 2. Create Kernels using Outer Product
        # Gx (Horizontal) detects vertical edges
        sobel_x = np.outer(s, d)
        
        # Gy (Vertical) detects horizontal edges
        sobel_y = np.outer(d, s)
        
        return sobel_x, sobel_y
    
    sobel_horizontal, sobel_vertical = get_sobel_kernels(kernel_size)
    
    if direction == "h": 
        horizontal_gradient = conv2D(img, sobel_horizontal, hstep , vstep)
        return horizontal_gradient
    
    elif direction == "v": 
        vertical_gradient = conv2D(img, sobel_vertical, hstep, vstep)
        return vertical_gradient
    
    elif direction == "both": 
        horizontal_gradient = conv2D(img, sobel_horizontal, hstep, vstep)
        vertical_gradient = conv2D(img, sobel_vertical, hstep, vstep)
        grad= np.sqrt(horizontal_gradient **2 + vertical_gradient **2)
        grad = (grad / grad.max() * 255)
        return grad.astype(np.uint8)
       
       
def contour_extractor(img : np.ndarray = None, save : bool = False, filename : str | None = None)->np.ndarray:
    '''
    Docstring for contour_extractor : 
    
        extract the contour of the thresholded image, where interior is white and background is black 
        
    :param img: image as np.ndarray 
    :type img: np.ndarray
    :return: contour image 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
    
    if img.ndim == 3: 
        raise ValueError("Please enter a binary image only")
    
    dx = [-1,-1,0,1,1,1,0,-1]
    dy = [0,-1,-1,-1,0,1,1,1]
    
    height, width = img.shape
    padded_arr = np.pad(img, pad_width=1, mode='constant', constant_values=255)
    has_zero_neighbor = np.zeros((height, width), dtype=bool)
    
    for k in range(8):
        r_start = 1 + dx[k]
        r_end   = 1 + dx[k] + height
        c_start = 1 + dy[k]
        c_end   = 1 + dy[k] + width
        
        neighbor_view = padded_arr[r_start:r_end, c_start:c_end]
        has_zero_neighbor |= (neighbor_view == 0)
    
    contour_mask = (img == 255) & has_zero_neighbor
    contour = np.argwhere(contour_mask)
    contour_list = [tuple(x) for x in contour]
    
    '''Display the contour and save it, if asked'''
    
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    row, col = zip(*contour_list)
    ax.scatter(col, row, s=2, c="red")
    ax.axis("off")
    plt.show()
    
    
    contour_img = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8); 
    for (i, j) in contour_list: 
        contour_img[i, j] = 255

    if save:
        if filename is  None:      
            raise ValueError("You want to save the image but did not provide a filename with extension")
        else:
            # creating new gray scale figure for saving 
             
            fig2, ax2 = plt.subplots()
            ax2.imshow(contour_img, cmap="gray")
            ax2.axis("off")
            fig2.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig2)

    return contour_img
