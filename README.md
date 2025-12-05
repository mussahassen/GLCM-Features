# Haralick Texture Features

A Python implementation of Haralick texture features for image analysis, based on the paper "Gray-level invariant Haralick texture features" by Tommy Löfstedt, Patrik Brynolfsson, Thomas Asklund, Tufve Nyholm, and Anders Garpebring.

These features are widely used in medical imaging, remote sensing, and computer vision for texture analysis and classification. This project provides a pipeline for extracting Haralick texture features from grayscale images using Gray Level Co-occurrence Matrices (GLCM). In this analysis I delve into the characteristics of these features. Characteristics include robustness; variance (rotation, gray level, image scale), or what features they may highlight from an image. This is done initially by applying texture features to randomly genrated toy images.

### Features included:
- Autocorrelation
- Cluster Prominence
- Cluster Shade
- Contrast
- Correlation 
- Difference Entropy 
- Difference Variance
- Dissimilarity 
- Energy 
- Entropy 
- Homogeneity
- Inverse Difference
- Sum Average 
- Sum Entropy 
- Sum of Squares 
- Sum Variance


The experimental design involves generating $8 \times 8$ binary images where each pixel is an independent and identically distributed (i.i.d.) Bernoulli random variable (only 2 gray levels, which makes for simple GLCM matricies). Nine distinct Bernoulli probabilities, $p \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$, are used to cover a variety of image outcomes...