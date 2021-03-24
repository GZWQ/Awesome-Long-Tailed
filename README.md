# Awesome-Long-Tailed
Papers about long-tailed tasks



## Re-weighting



## Sampling



## Data Augmentation



[Feature Space Augmentation for Long-Tailed Data](https://arxiv.org/pdf/2008.03673.pdf) ECCV 2020

[M2m: Imbalanced Classification via Major-to-minor Translation](https://arxiv.org/pdf/2004.00431.pdf) CVPR 2020

[Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective](https://arxiv.org/pdf/2002.10826.pdf) CVPR 2020

[Feature Transfer Learning for Deep Face Recognition with Under-Represented Data](https://arxiv.org/pdf/1803.09014.pdf) CVPR 2019

[Memory-based Jitter: Improving Visual Recognition on Long-tailed Data with Diversity In Memory](https://arxiv.org/pdf/2008.09809.pdf) preprint

[FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation](https://arxiv.org/pdf/2102.12867.pdf)

 preprint

> 1. Augmentation is performed in the feature space.
> 2. For a class to be augmentated, we can compute its mean features from current batch data and then update it in a momentum way. So we can generate virtual features by firstly randomly sampling variation from a gaussian prior and add the variation to the mean feature.