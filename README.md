# Haralick Texture Features

A Python implementation of Haralick texture features for image analysis, based on the paper "Gray-level invariant Haralick texture features" by Tommy Löfstedt, Patrik Brynolfsson, Thomas Asklund, Tufve Nyholm, and Anders Garpebring.

These features are widely used in medical imaging, remote sensing, and computer vision for texture analysis and classification. This project provides a pipeline for extracting Haralick texture features from grayscale images using Gray Level Co-occurrence Matrices (GLCM). In this analysis I delve into the characteristics of these features. Characteristics include robustness; variance (rotation, gray level, image scale), or what features they may highlight from an image. This is done initially by applying texture features to randomly genrated toy images.

### Features included:
- Autocorrelation
- Cluster Prominence
- Cluster Shade
- Cluster Tendency
- Contrast
- Correlation 
- Difference Average
- Difference Entropy 
- Difference Variance
- Energy 
- Entropy 
- Information measure of correlation 1
- Information measure of correlation 2
- Inverse Difference
- Inverse Difference Moment
- Inverse Variance
- Joint Average
- Maximal Correlation Coefficient
- Maximum Probability
- Sum Average 
- Sum Entropy 
- Sum of Squares 


The experimental design involves generating $8 \times 8$ binary images where each pixel is an independent and identically distributed (i.i.d.) Bernoulli random variable (only 2 gray levels, which makes for simple GLCM matricies). Nine distinct Bernoulli probabilities, $p \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$, are used to cover a variety of image outcomes...

## Guide

Start with the 'Htf.pdf' file to go over notation and the list of all the features. 

The main experiments can be found in 'Experiments.ipynb' and 'Experiments_p2.ipynb'. In 'feature_char.ipynb', there is an introduction to the charecterization of the features.

In 'toy_images.ipynb' you can find $8 \times 8$ generated images (more complex images will be generated), to get a visual sense of the distinct textures that the features measure. 'Validation.ipynb' consists of executions of the functions as a sample test. 'features.py' under the src folder holds all the functions for the new features.