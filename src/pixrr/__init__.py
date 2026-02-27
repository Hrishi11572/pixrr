# peda_img/__init__.py

from .edges import gradient_prewitt, gradient_sobel, contour_extractor
from .enhance import linear_contrast_enhancement, histogram_equalization
from .filters import padd_image, conv2D, laplacian, sharpen_image, gaussian_filter, gaussian_smoothing
from .io import handle_image, convert_to_gray, show_image, save_image, plot_img_hist
from .threshold import threshold_image, otsu_thresholding
from .segmentation import kmeans_segmentation

# 2. DEFINING EXPORTS
# This list controls what happens if someone types "from peda_img import *"
# It also cleans up the namespace so IDEs know what is public.

__all__ = [
    'gradient_prewitt',
    'gradient_sobel',
    'contour_extractor',
    'linear_contrast_enhancement',
    'histogram_equalization',
    'padd_image',
    'conv2D',
    'laplacian',
    'sharpen_image',
    'gaussian_filter',
    'gaussian_smoothing',
    'handle_image',
    'convert_to_gray',
    'show_image',
    'save_image',
    'plot_img_hist',
    'threshold_image',
    'kmeans_segmentation',
    'otsu_thresholding'
]

# Optional: Library Metadata
__version__ = "0.1.0"
__author__ = "Hrishikesh Tiwari"