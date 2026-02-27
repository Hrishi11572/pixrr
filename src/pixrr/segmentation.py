import statistics
import numpy as np 
import matplotlib.pyplot as plt 
from .io import convert_to_gray

    
def kmeansOnHistogram(img: np.ndarray, k : int = 2)->np.ndarray: 
    '''
    Docstring for kmeansOnHistogram
    
    :param img: the image as numpy array 
    :type img: np.ndarray
    :param k: the number of clusters
    :type k: int
    :return: the image, as numpy array 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
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

def kmeans_segmentation(img:np.ndarray, k: int = 2, iterations: int = 5, save : bool = False , filename : str | None = None)->np.ndarray:
    '''
    Docstring for kmeans_segmentation
    
    :param img: input image as a numpy array 
    :type img: np.ndarray
    :param k: the number of clusters 
    :type k: int
    :param iterations: how many times the algorithm must run
    :type iterations: int
    :param save: if you want to save the image
    :type save: bool
    :param filename: the name of the file, where you want to save the image
    :type filename: str | None
    :return: a numpy array 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
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