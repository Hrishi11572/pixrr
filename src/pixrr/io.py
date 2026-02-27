from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 


def handle_image(filepath:str)->np.ndarray: 
    '''
    Job : load the image into a consistent, usable data-structure for processing
        
    - opens image using pillow 
    - convert image to array. 
    '''
    
    # input the image as a pillow object 
    try: 
        with Image.open(filepath) as pil_img_obj:
            # convert the image to numpy array
            npimage = np.array(pil_img_obj).astype(np.uint8)
    except Exception as e : 
            raise IOError(f"Failed to load image {filepath} : {e}")
        
    
    # Case 1: Pure grayscale (H, W)
    if npimage.ndim == 2: 
        return npimage
    
     # Case 2: (H, W, 1) grayscale with extra channel dim
    if npimage.ndim == 3 and npimage.shape[2] == 1:
        return npimage[:, :, 0]
    
    # Case 3: RGB (H, W, 3)
    if npimage.ndim == 3 and npimage.shape[2] == 3:
        return npimage

    # Case 4: RGBA (H, W, 4) → drop alpha
    if npimage.ndim == 3 and npimage.shape[2] == 4:
        return npimage[:, :, :3]
    
    # Otherwise, unsupported
    raise ValueError(
        f"Unsupported image shape {npimage.shape}. Expected grayscale or RGB/RGBA."
    )


def convert_to_gray(img : np.ndarray | None = None)->np.ndarray:
    '''
    Docstring for convertToGray
    
    :param img: Input the image as np.ndarray 
    :type img: np.ndarray | None
    :return: returns the gray-scale image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
    if img is None: 
        raise ValueError("Please enter an Image as np.ndarray")
    
    if img.ndim == 2: # already gray 
        print("The input image is already gray-scale")
        return img

    # Grayscale with redundant channel: (H, W, 1)
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    
    # Proper RGB image: (H, W, 3)
    if img.ndim == 3 and img.shape[2] == 3:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.uint8)

    # Cannot convert arbitrary multi-channel images
    raise ValueError(f"Cannot convert image with shape {img.shape} to grayscale.")


def show_image(img : np.ndarray, channel : str = "all")->None: 
    '''
    Docstring for showImage
    
    :param img: a numpy array 
    :type img: np.ndarray
    
    - Takes an image as input as an np.ndarray 
    - Displays the image on the console. 
    '''
    if img.ndim == 3: 
        if channel == "all": 
            plt.axis("off")
            plt.imshow(img)
            plt.show()
        else :
            def display_colored(img: np.ndarray, level:int)->None: 
                if level >= 3: 
                    raise ValueError("No such channel exists")
                else: 
                    newimg = np.zeros_like(img)
                    newimg[:,:,level] = img[:,:,level]
                    plt.axis("off")
                    plt.imshow(newimg)
                    plt.show()
            
            if channel == "r": 
                display_colored(img, 0)
            if channel == "g": 
                display_colored(img, 1)
            if channel == "b": 
                display_colored(img, 2)
    else: 
        plt.axis("off")
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.show()
        plt.close()
        

def save_image(img: np.ndarray, filename:str = "default.png")->None:
    '''
    Docstring for saveImage
    
    :param img: a numpy array
    :type img: np.ndarray

    - Saves the image 
    '''
    
    Image.fromarray(img).save(filename)
    return None 



def plot_img_hist(img : np.ndarray, channel : str ="gray", curve_type="boxy" ,save: bool = False, filename : str | None = None)->None: 
    '''
    Docstring for plot_img_hist
    
    :param img: A numpy array of shape (H,W,C)
    :type img: np.ndarray
    :param channel: the channels "gray", "red", "green", "blue", "all" (to see all three in one image)
    :type channel: str
    :param save: make it True if you want to save this histogram image in your Current Directory.
    :type save: bool
    :param curve_type : does the user want to view bar graph style histogram or continuous style histogram 
    :type curve_type : str
    '''

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
    
    if img.ndim == 2: 
        ''' the user wants to see the gray scale histogram of a gray scale image '''
        if channel != "gray": 
            raise ValueError("Cannot request RGB histogram from a grayscale image.")
        else: 
            plot_boxy_histogram(img=img,t=(),save=save,filename=filename) if curve_type == "boxy" else plot_smooth_histogram(img=img,t=(),save=save,filename=filename)
    elif img.ndim == 3: 
        ''' it is an RGB image, and the user might want to see any type of histogram'''
        if channel == "gray": 
            red = img[:,:,0]
            green = img[:,:,1]
            blue = img[:,:,2]
            gray = (0.299 * red + 0.587 * green + 0.114 * blue).astype(np.uint8)
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


if __name__ == "__main__":
    # Example 
    img = handle_image("testfile.png")
    gray_img = convert_to_gray(img)
    save_image(gray_img, "oppenheimer.png")