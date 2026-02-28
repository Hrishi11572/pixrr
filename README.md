# **pixrr : A lightweight image processing toolkit for python**

`pixrr` is a lightweight, beginner-friendly image processing library built for fast experimentation and teaching.
It focuses on simplicity, clean function names, and easy-to-understand code, making it useful both for quick image tasks and for pedagogical environments such as introductory image processing courses.

---

##  Features (Planned & Implemented)

### **Core Utilities**

* Convert color images to grayscale and binary
* Basic I/O helpers (read, write, display)
* Plot and analyze histograms
* Image thresholding
* Image Cropping and Rotation (planned)

### **Filtering & Enhancement**

* Convolution using masks
* Convolution using FFT (planned)
* Contrast enhancement
* Histogram equalization
* Smoothing filters (Gaussian, median, weighted median)
* Sharpening and Laplacian operators

### **Edge and Gradient Detection**

* Prewitt and Sobel (horizontal & vertical)
* Second-order derivatives
* Canny edge detector (planned)

### **Segmentation & Classification**

* Otsu thresholding 
* K-means clustering for image segmentation
* Mean-Shift clustering (planned)
* EM/Bayesian pixel classification (planned)

---

## Project Roadmap

| Stage  | Goal                                                    |
| ------ | ------------------------------------------------------- |
| Part 1 | Basic conversions, contour extraction, histogram tools  |
| Part 2 | Convolution, enhancement techniques, gradient operators |
| Part 3 | Automatic segmentation & clustering                     |
| Part 4 | Advanced edge detection and EM classification           |

---

## Installation (when it’s live)

```bash
pip install pixrr
```

*(Currently under development. Not on PyPI yet.)*

---

## Usage Example

```python
import pixrr as pix

# Example 
img = pix.handle_image("peda_img/test_images/test2.png")
grey = pix.convert_to_gray(img)
# Applying gaussian
pix.show_image(pix.gaussian_smoothing(grey, kernel_size=3))
```

---

## Contribution

This library is early-stage but open to improvements, bug reports, and algorithm implementations.
Ideal for students learning image processing or developers wanting simple utilities without heavy dependencies.

---

