import numpy as np 
import matplotlib.pyplot as plt 
from pixrr.io import handle_image, convert_to_gray, show_image, save_image
import statistics
from numba import njit 

def plot_img_hist(img : np.ndarray, channel : str ="all", curve_type="boxy" ,save: bool = False, filename : str | None = None)->None: 
    def plot_boxy_histogram(img : np.ndarray, t : tuple = (), save: bool = False, filename : str | None = None) -> None:
        colors = ("red", "green", "blue", "gray")
        
        fig, ax = plt.subplots()
        ax.set_xlim([0, 255])
        
        ax.set_xlabel("Intensity value")
        ax.set_ylabel("pixel count")    
        
        if len(t) == 0: 
            # asking me to plot grayscale histogram  
            ax.set_title("Grayscale Histogram")
            plt.hist(img.flatten(), bins = 256, range=(0,255),color=colors[3])
        elif len(t) == 1: 
            ax.set_title(f"{colors[t[0]]} Histogram")
            plt.hist(img[:,:,t[0]].flatten(), bins = 256, range=(0,255),color=colors[t[0]])
        elif len(t) == 3: 
            ax.set_title("Color Histogram")
            for i in range(3): 
                plt.hist(img[:,:,t[i]].flatten(), bins=256, range=(0,255), color=colors[t[i]])
        plt.show()
        
        ''' Save the image if asked to '''
        if save and filename is not None: 
            fig.savefig(filename, dpi=300)
            
        return None 
    
    def plot_smooth_histogram(img : np.ndarray, t : tuple = (), save : bool = False, filename : str | None = None) -> None:
        colors = ("red", "green", "blue")
        fig, ax = plt.subplots()
        ax.set_xlim([0, 255])
        
        if len(t) == 0: 
            # asking to plot gray scale 
            ax.set_title("Grayscale Histogram")
            hist, bin_edges_ = np.histogram(img, bins=256, range=(0,255))
            ax.plot(bin_edges_[:-1], hist, color = 'gray')
        else:   
            for channelID, color in enumerate(colors): 
                if channelID in t: 
                    hist, bin_edges_ = np.histogram(img[:,:,channelID], bins=256, range=(0,255))
                    ax.plot(bin_edges_[:-1], hist, color = color)

            if len(t) == 1: 
                ax.set_title(f"{colors[t[0]]} Histogram")
            elif len(t) == 3: 
                ax.set_title("Color Histogram")    
                
        ax.set_xlabel("Intensity value")
        ax.set_ylabel("pixel count")    
        plt.show()
        
        ''' Save the image if asked to '''
        if save and filename is not None: 
            fig.savefig(filename,dpi=300)
            
        return None 
        
    if img.ndim == 2 and channel != "gray":
        raise ValueError("Cannot request RGB histogram from a grayscale image.")
  
    if channel == "gray": 
        if img.ndim == 3: 
            red = img[:,:,0]
            green = img[:,:,1]
            blue = img[:,:,2]
            gray = (0.299 * red + 0.587 * green + 0.114 * blue).astype(np.uint8)
        else: 
            gray = img  
        plot_boxy_histogram(img=gray,t=(),save=save,filename=filename) if curve_type == "boxy" else plot_smooth_histogram(img=gray,t=(),save=save,filename=filename)
        
    elif channel == "r": 
        plot_boxy_histogram(img, t=(0,), save=save, filename=filename) if curve_type == "boxy" else plot_smooth_histogram(img, t=(0,), save=save, filename=filename)
    elif channel == "g":
        plot_boxy_histogram(img, t=(1,), save=save, filename=filename) if curve_type == "boxy" else plot_smooth_histogram(img, t=(1,), save=save, filename=filename)
    elif channel == "b": 
        plot_boxy_histogram(img, t=(2,), save=save, filename=filename) if curve_type == "boxy" else plot_smooth_histogram(img, t=(2,), save=save, filename=filename)
    elif channel == "all":
        plot_boxy_histogram(img, t=(0,1,2), save=save, filename=filename) if curve_type == "boxy" else plot_smooth_histogram(img, t=(0,1,2), save=save, filename=filename)
  
    return None



def threshold_image(img: np.ndarray | None = None, thresholdValue : int = 0, inverse : bool = False)->np.ndarray:
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



def contourExtractor(img : np.ndarray = None, save : bool = False, filename : str | None = None)->np.ndarray:
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



def kmeans_segmentation(img:np.ndarray, k: int = 2, iterations: int = 5, save : bool = False , filename : str | None = None)->np.ndarray:
    labels = []
    
    for i in range(iterations):
        labels.append(kmeansOnHistogram(img, k))
    
    transposed_lables = list(zip(*labels))
    
    result_list = [statistics.mode(column) for column in transposed_lables]

    def get_nice_colors(k, palette="Set3"):
        cmap = plt.get_cmap(palette)
        colors = cmap(np.linspace(0, 1, k))[:, :3]  # drop alpha
        colors = (colors * 255).astype(np.uint8)
        return [tuple(col) for col in colors]
    
    colors = get_nice_colors(k, palette="Accent")  

    color_arr = np.array(colors, dtype=np.uint8)
    new_img = color_arr[result_list].reshape(img.shape[0], img.shape[1], 3)

    plt.axis("off")
    plt.imshow(new_img)
    plt.show()
    

    if save: 
        if filename is None: 
            raise ValueError("You want to save the image but did not provide a filename with extension")
        else:
            # 1. Create a new figure and axes for saving
            fig2, ax2 = plt.subplots()
            
            # 2. Draw the new image onto the *saving* axes
            ax2.imshow(new_img)
            
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
    


def kmeansOnHistogram(img: np.ndarray, k : int = 2)->np.ndarray: 
    if k == 0 or k == 1: 
        raise ValueError("Number of clusters (k) should be greater than or equal to 2")
    
    if img.ndim == 3: 
        img = convert_to_gray(img) 
    else: 
        img = img 

    hist, _ = np.histogram(img, bins=256, range=(0,255))
    
    def kmeans_plus_plus(hist : np.ndarray , k : int)->np.ndarray:
        means = []
        
        first_center = np.random.choice(np.arange(256), p=hist/hist.sum())
        means.append(first_center)
        
        for j in range(1 ,k):
            distribution = []
            for x in range(256) :                 
                min_dist2 = min((x - mean)**2 for mean in means)
                distribution.append(min_dist2 * hist[x])
                
            distribution = np.array(distribution)
            if distribution.sum() == 0: 
                continue
            
            new_center = np.random.choice(256, p=distribution/distribution.sum())
            means.append(new_center)
        
        means = np.array(means)
        return np.sort(means)
    
    
    ls = kmeans_plus_plus(hist=hist, k=k) # <-- array of k centers 
    newls = ls.copy()
    
    tolerance = 1 
    
    while True: 
        # Run the iteration 
        ls = newls.copy()
        
        for i in range(ls.shape[0]):
            if i-1 >= 0 and i+1 < ls.shape[0]:
                sum_1 = 0 
                sum_2 = 0
                for j in range((ls[i-1] + ls[i])//2, (ls[i] + ls[i+1])//2 + 1): 
                    sum_1 += j * hist[j]
                    sum_2 += hist[j]
                
                if sum_2 == 0: 
                    continue
                
                newls[i] = sum_1/sum_2 
        
        # Check Convergence Criteria here                 
        if np.max(np.abs(ls - newls)) < tolerance: 
            break 
    
    thresholds = np.array([(ls[i] + ls[i+1]) // 2 for i in range(len(ls) - 1)])
    flat = img.reshape(-1)

    labels = np.zeros(flat.shape, dtype=np.int32)

    # first cluster
    labels[flat <= thresholds[0]] = 0

    # middle clusters
    for i in range(1, len(ls) - 1):
        labels[(flat > thresholds[i-1]) & (flat <= thresholds[i])] = i

    # last cluster (the one you forgot)
    labels[flat > thresholds[-1]] = len(ls) - 1

    return labels



def linear_contrast_enhancement(img : np.ndarray, low: int = 0, high : int = 255 , save : bool = False, filename : str | None = None)->np.ndarray:
    
    if img.ndim == 3: 
         # convert colored to gray scale 
         img = convert_to_gray(img)
    
    if not (0 <= low < high <= 255):
        raise ValueError(f"{low} and {high} must satisfy 0 <= low < high <= 255")

    
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


''' Morphological operations in python, all by myself '''

''' Covolution '''

# @njit
# def paddImage(img: np.ndarray, row_pad : int = 1 , col_pad: int = 1):
#     if img.ndim == 3: 
#         # RGB image 
#         padded_img = np.pad(img, pad_width=((row_pad, row_pad), (col_pad, col_pad), (0, 0)), mode="constant", constant_values=0)
#     else: 
#         # gray-scale image 
#         padded_img = np.pad(img, pad_width=((row_pad, row_pad), (col_pad, col_pad)), mode="constant", constant_values=0)
    
#     return padded_img

''' because the above version of paddImage was incompatible with numba jit '''

@njit
def padd_image(img: np.ndarray, row_pad: int = 1, col_pad: int = 1):
    # Support both grayscale (2D) and RGB (3D) images in numba-friendly way
    if img.ndim == 2:
        h, w = img.shape
        ph, pw = h + 2*row_pad, w + 2*col_pad
        padded = np.zeros((ph, pw), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                padded[i + row_pad, j + col_pad] = img[i, j]
        return padded
    elif img.ndim == 3:
        h, w, ch = img.shape
        ph, pw = h + 2*row_pad, w + 2*col_pad
        padded = np.zeros((ph, pw, ch), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                for k in range(ch):
                    padded[i + row_pad, j + col_pad, k] = img[i, j, k]
        return padded
    else:
        raise ValueError("Unsupported image dimensions for paddImage")

@njit
def conv2D(img: np.ndarray, mask : np.ndarray, hstep: int = 1, vstep: int = 1)->np.ndarray: 
 
    if not (mask.shape[0] % 2 == 1 and mask.shape[1]%2 == 1): 
        raise ValueError("mask should have sape of the form : (odd_val, odd_val)")
    
    padded_img = padd_image(img, mask.shape[0]//2 , mask.shape[1]//2)
    
    
    if img.ndim == 2: 
        kernel_height, kernel_width = mask.shape
        padded_height, padded_width = padded_img.shape 
        
        output_height = (padded_height - kernel_height) // vstep + 1
        output_width = (padded_width - kernel_width) // hstep + 1
        
        new_img = np.zeros((output_height, output_width), dtype="float64")
        mask = mask[::-1, ::-1]
        
        for y in range(0, output_height): 
            for x in range(0, output_width): 
                region = padded_img[y*vstep: y*vstep + kernel_height, x*hstep : x*hstep + kernel_width]
                new_img[y][x] = np.sum(region * mask)
        
        return new_img
    
    elif img.ndim == 3: 
        channels = []
        mask = mask[::-1, ::-1]

        for i in range(3):
            channel = conv2D(img[:,:,i], mask, hstep, vstep)
            channels.append(channel)
        
        # return np.stack(channels, axis=2)
        h, w = channels[0].shape
        c = len(channels)
        out = np.zeros((h, w, c), dtype=channels[0].dtype)
        for k in range(c):
            out[:, :, k] = channels[k]
        return out
    else: 
        raise ValueError("Unsupported image type!")
    

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
       

@njit
def laplacian(img : np.ndarray)->np.ndarray:
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
    img_F = img.astype(np.float32)
    lap = laplacian(img)
    
    sharpImage = img_F - c * lap 
    sharpImage = np.clip(sharpImage, 0, 255)
    
    return sharpImage.astype(np.uint8)
    # return (img - c * laplacian(img)).astype(np.uint8) -- without clipping is bad



@njit
def gaussian_filter_kernel(sigma: float = 1.0, size: int = 3): 
    '''
        https://stackoverflow.com/a/43346070
    '''
    
    if size % 2 == 0: 
        raise ValueError("Kernel sizes should be odd")
    
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def gaussian_smoothing(img : np.ndarray, kernel_size: int = 3, sigma : float = 1.0)->np.ndarray:
    gauss = gaussian_filter_kernel(sigma, kernel_size)
    
    smoothed = conv2D(img, gauss)
    smoothed = np.clip(smoothed, 0, 255)
    
    return smoothed.astype(np.uint8)


if __name__ == "__main__": 
    # Example 
    img = handle_image("peda_img/test_images/test2.png")
    grey = convert_to_gray(img)
    # applying laplacian
    show_image(gaussian_smoothing(grey, kernel_size=3))
    
    
'''
shape = (836, 906, 3)

array(
  [
    [ [R,G,B], [R,G,B], [R,G,B], ..., [R,G,B] ],   <-- 906 pixels in this row
    [ [R,G,B], [R,G,B], [R,G,B], ..., [R,G,B] ],
    ...
    836 such rows
  ]
)
'''

'''
np.clip(arr, min_val, max_val)
For every element in arr:

if it’s less than min_val, replace it with min_val

if it’s greater than max_val, replace it with max_val

otherwise, leave it alone



If you don’t clip and jump straight to uint8:

(-23.7).astype(np.uint8) → 233
(412.9).astype(np.uint8) → 156

'''