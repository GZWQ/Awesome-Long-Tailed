



# Bechmark



## CIFAR10_LT

Using ResNet-32 as backbone.

Imbalance factor = 100

|                         |  Acc  |
| :---------------------: | :---: |
|     BBN (CVPR 2020)     | 79.82 |
|     RSG (CVPR 2021)     | 79.95 |
|  MetaSAug (CVPR 2021)   | 80.66 |
| LogitAdjust (ICLR 2021) | 80.92 |
|     ACE (ICCV 2021)     | 81.4  |
|   RIDE (ICLR 2021) *    | 81.54 |
|   MiSLAS (CVPR 2021)    | 82.1  |
|    BLMS (CVPR 2020)     | 84.9  |

> \* denotes results reproduced by me.

## CIFAR100_LT

Using ResNet-32 as backbone.

Imbalance factor = 100

|                         | Many | Medium | Few  |  All  |
| :---------------------: | :--: | :----: | :--: | :---: |
|       BBN (CVPR)        |      |        |      | 42.56 |
| LogitAdjust (ICLR 2021) |  -   |   -    |  -   | 43.89 |
|     RSG (CVPR 2021)     |  -   |   -    |  -   | 44.55 |
|    LADE (CVPR 2021)     |      |        |      | 45.4  |
|     SSD (ICCV 2021)     |      |        |      | 46.0  |
|   MiSLAS (CVPR 2021)    |      |        |      | 47.0  |
|  MetaSAug (CVPR 2021)   |  -   |   -    |  -   | 48.01 |
|    RIDE (ICLR 2021)     | 69.3 |  49.3  | 26.0 | 49.1  |
|     ACE (ICCV 2021)     | 66.3 |  52.8  | 27.2 | 49.6  |
|       TADE (2021)       |      |        |      | 49.8  |
|    BLMS (CVPR 2020)     |      |        |      | 50.8  |
|    PaCo (ICCV 2021)     |  -   |   -    |  -   | 52.0  |



## Places365_LT

Using ResNet-152 as backbone.

|                     | Many | Medium | Few  | All  |
| :-----------------: | :--: | :----: | :--: | :--: |
|  LADE (CVPR 2021)   | 42.8 |  39.1  | 29.6 | 38.8 |
|    LDA (MM 2021)    | 41.0 |  40.7  | 32.1 | 39.1 |
|   RSG (CVPR 2021)   | 41.9 |  41.4  | 32.0 | 39.3 |
| GistNet (ICCV 2021) | 42.5 |  40.8  | 32.1 | 39.6 |
| MiSLAS (CVPR 2021)  |      |        |      | 40.4 |
|     TADE (2021)     |      |        |      | 40.9 |
|  PaCo (ICCV 2021)   | 36.1 |  47.9  | 35.3 | 41.2 |



## ImageNet_LT

Using ResNet-50 as backbone.

|                         | Many | Medium | Few  |  All  |
| :---------------------: | :--: | :----: | :--: | :---: |
|  MetaSAug (CVPR 2021)   |      |        |      | 47.39 |
| LogitAdjust (ICLR 2021) |      |        |      | 51.1  |
|     KCL (ICLR 2021)     | 61.8 |  49.4  | 30.9 | 51.5  |
|   MiSLAS (CVPR 2021)    |  -   |   -    |  -   | 52.7  |
|      LDA (MM 2021)      | 64.5 |  50.9  | 31.5 | 53.4  |
|     ACE (ICCV 2021)     |  -   |   -    |  -   | 54.7  |
|    RIDE (ICLR 2021)     |  -   |   -    |  -   | 55.4  |
|       CBD (2021)        | 68.5 |  52.7  | 29.2 | 55.6  |
|     SSD (ICCV 2021)     | 66.8 |  53.1  | 35.4 | 56.0  |
|       MoE (2021)        | 66.7 |  54.1  | 37.6 | 56.7  |
|    PaCo (ICCV 2021)     |  -   |   -    |  -   | 57.0  |
|       TADE (2021)       | 66.5 |  57.0  | 43.4 | 58.8  |



## iNaturalist18

Using ResNet-50 as backbone.

|                         | Many | Medium | Few  |  All  |
| :---------------------: | :--: | :----: | :--: | :---: |
| LogitAdjust (ICLR 2021) |  -   |   -    |  -   | 66.36 |
|  MetaSAug (CVPR 2021)   |  -   |   -    |  -   | 68.75 |
|     KCL (ICLR 2021)     |  -   |   -    |  -   | 68.6  |
|     BBN (CVPR 2020)     |  -   |   -    |  -   | 69.62 |
|    LADE (CVPR 2021)     |  -   |   -    |  -   | 70.0  |
|     RSG (CVPR 2021)     |  -   |   -    |  -   | 70.26 |
|   GistNet (ICCV 2021)   |      |        |      | 70.8  |
|     SSD (ICCV 2021)     |      |        |      | 71.5  |
|   MiSLAS (CVPR 2021)    |  -   |   -    |  -   | 71.6  |
|    RIDE (ICLR 2021)     | 70.9 |  72.4  | 73.1 | 72.6  |
|     ACE (ICCV 2021)     |      |        |      | 72.9  |
|       TADE (2021)       |  -   |   -    |  -   | 72.9  |
|    PaCo (ICCV 2021)     |  -   |   -    |  -   | 73.2  |
|       CBD (2021)        | 75.9 |  74.7  | 71.5 | 73.6  |
|       MoE (2021)        | 72.8 |  74.8  | 74.6 | 74.5  |



# Awesome-Long-Tailed

Papers related to long-tailed tasks



## Survey

[Deep Long-Tailed Learning: A Survey](https://arxiv.org/pdf/2110.04596.pdf)  

[Bag of Tricks for Long-Tailed Visual Recognition with Deep Convolutional Neural Networks](http://www.lamda.nju.edu.cn/zhangys/papers/AAAI_tricks.pdf) (AAAI 2021) [Code](https://github.com/zhangyongshun/BagofTricks-LT)  



## Re-weighting

[Influence-Balanced Loss for Imbalanced Visual Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Influence-Balanced_Loss_for_Imbalanced_Visual_Classification_ICCV_2021_paper.pdf) (ICCV 2021) [Code](https://github.com/pseulki/IB-Loss) 

[Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection](https://arxiv.org/pdf/2012.08548.pdf) (CVPR 2021) [Code](https://github.com/tztztztztz/eqlv2) 

>  Gradient-guided re-weighting

[Adaptive Class Suppression Loss for Long-Tail Object Detection](https://arxiv.org/pdf/2104.00885.pdf) (CVPR 2021) [Code](https://github.com/CASIA-IVA-Lab/ACSL) 

(LADE) [Disentangling Label Distribution for Long-tailed Visual Recognition](https://arxiv.org/pdf/2012.00321.pdf) (CVPR 2021) [Code](https://github.com/hyperconnect/LADE) 

> An extension of BalancedSoftmax while improvement is quite limited. But it can handle unknown distributions of testing datasets.

(LogitAdjust) [Long-tail learning via logit adjustment](https://arxiv.org/pdf/2007.07314v1.pdf) (ICLR 2021)

(BLMS) [Balanced Meta-Softmax for Long-Tailed Visual Recognition](https://papers.nips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf) (NIPS 2020) [code](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification) 

## Sampling



## Meta Learning

(MetaSAug) [MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2103.12579.pdf) (CVPR 2021) [code](https://github.com/BIT-DA/MetaSAug) 

[Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition from a Domain Adaptation Perspective](https://arxiv.org/pdf/2003.10780.pdf) (CVPR 2020) [code](https://github.com/abdullahjamal/Longtail_DA) 

[Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://arxiv.org/pdf/1902.07379.pdf) (NIPS 2019) [code](https://github.com/xjtushujun/Meta-weight-net_class-imbalance) 







## Feature Manipulation

(SSD) [Self Supervision to Distillation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2109.04075.pdf) [ICCV 2021] [waiting code](https://github.com/MCG-NJU/SSD-LT) 

[Procrustean Training for Imbalanced Deep Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Procrustean_Training_for_Imbalanced_Deep_Learning_ICCV_2021_paper.pdf) (ICCV 2021) 

[Self Supervision to Distillation for Long-Tailed Visual Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Self_Supervision_to_Distillation_for_Long-Tailed_Visual_Recognition_ICCV_2021_paper.pdf) (ICCV 2021) 

(GistNet) [GistNet: a Geometric Structure Transfer Network for Long-Tailed Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_GistNet_A_Geometric_Structure_Transfer_Network_for_Long-Tailed_Recognition_ICCV_2021_paper.pdf) (ICCV 2021) 

(PaCo) [Parametric Contrastive Learning](https://arxiv.org/pdf/2107.12028.pdf)  (ICCV 2021) [Code](https://github.com/dvlab-research/Parametric-Contrastive-Learning)

(RSG) [RSG: A Simple but Effective Module for Learning Imbalanced Datasets](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_RSG_A_Simple_but_Effective_Module_for_Learning_Imbalanced_Datasets_CVPR_2021_paper.pdf) (CVPR 2021) [Code](https://github.com/Jianf-Wang/RSG) 

> RSG(Rare-class Sample Generator) aims to generate some new samples for rare classes during training.

[Contrastive Learning based Hybrid Networks for Long-Tailed Image Classification](https://arxiv.org/pdf/2103.14267.pdf) (CVPR 2021) [Waiting code](https://paperswithcode.com/paper/contrastive-learning-based-hybrid-networks) 

(MiSLAS) [Improving Calibration for Long-Tailed Recognition](https://arxiv.org/pdf/2104.00466.pdf) (CVPR 2021) [Code](https://github.com/Jia-Research-Lab/MiSLAS) 

(KCL) [Exploring Balanced Feature Spaces for Representation Learning](https://openreview.net/pdf?id=OqtLIabPTit) (ICLR 2021) [waiting code](https://paperswithcode.com/paper/exploring-balanced-feature-spaces-for)  

(SSP) [Rethinking the Value of Labels for Improving Class-Imbalanced Learning](https://arxiv.org/pdf/2006.07529.pdf) (NIPS 2020) [Code](https://github.com/YyzHarry/imbalanced-semi-self) 

[Feature Space Augmentation for Long-Tailed Data](https://arxiv.org/pdf/2008.03673.pdf) ECCV 2020

[Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500239.pdf) ECCV 2020 [Code](https://github.com/xiangly55/LFME) 

[M2m: Imbalanced Classification via Major-to-minor Translation](https://arxiv.org/pdf/2004.00431.pdf) CVPR 2020

(BBN) [Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition](https://arxiv.org/pdf/1912.02413.pdf) (CVPR 2020) [Code](https://github.com/Megvii-Nanjing/BBN) 

[Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective](https://arxiv.org/pdf/2002.10826.pdf) CVPR 2020

[Feature Transfer Learning for Deep Face Recognition with Under-Represented Data](https://arxiv.org/pdf/1803.09014.pdf) CVPR 2019

[Memory-based Jitter: Improving Visual Recognition on Long-tailed Data with Diversity In Memory](https://arxiv.org/pdf/2008.09809.pdf) preprint

[FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation](https://arxiv.org/pdf/2102.12867.pdf)  (ICCV 2021) [Code](https://github.com/yuhangzang/FASA)   

> 1. Augmentation is performed in the feature space.
> 2. For a class to be augmentated, we can compute its mean features from current batch data and then update it in a momentum way. So we can generate virtual features by firstly randomly sampling variation from a gaussian prior and add the variation to the mean feature.

(BKD) [Balanced Knowledge Distillation for Long-tailed Learning](https://arxiv.org/pdf/2104.10510.pdf) preprint [Code](https://github.com/EricZsy/BalancedKnowledgeDistillation) 

### Logit Adjustment

[Normalization Calibration (NorCal) for Long-Tailed Object Detection and Instance Segmentation](https://openreview.net/pdf?id=t9gKUW9T8fX)  (NIPS2021) [Code](https://github.com/tydpan/NorCal)  

[t-vMF Similarity for Regularizing Intra-Class Feature Distribution](https://staff.aist.go.jp/takumi.kobayashi/publication/2021/CVPR2021.pdf) (CVPR2021) [Code](https://github.com/tk1980/tvMF) 

[Distribution Alignment: A Unified Framework for Long-tail Visual Recognition](https://arxiv.org/pdf/2103.16370.pdf) (CVPR 2021) [Code](https://github.com/Megvii-BaseDetection/DisAlign) 

[Distilling Virtual Examples for Long-tailed Recognition](https://arxiv.org/pdf/2103.15042.pdf) (CVPR 2021)

> Disentangle the prediction from teach model into multiple "virtual examples", i.e., many "one-hot gt".

(LDA) [Long-tailed Distribution Adaptation](https://arxiv.org/pdf/2110.02686.pdf) (ACM MM 2021) [Code](https://github.com/pengzhiliang/LDA) 

## Multiple Experts

(ACE) [ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot](https://arxiv.org/pdf/2108.02385.pdf) (ICCV 2021) [Code](https://github.com/jrcai/ACE) 

(RIDE) [Long-tailed Recognition by Routing Diverse Distribution-Aware Experts](https://arxiv.org/pdf/2010.01809.pdf) (ICLR 2021) [Code](https://github.com/frank-xwang/RIDE-LongTailRecognition) 

(LDA) [Long-tailed Distribution Adaptation](https://arxiv.org/pdf/2110.02686.pdf) (ACM MM 2021) [Code](https://github.com/pengzhiliang/LDA) 

[Long-Tailed Recognition Using Class-Balanced Experts](https://arxiv.org/pdf/2004.03706.pdf) (DAGM-GCPR 2020) [Code](https://github.com/ssfootball04/class-balanced-experts) 

(CBD) [Class-Balanced Distillation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2104.05279.pdf) (preprint) [waiting code](https://paperswithcode.com/paper/class-balanced-distillation-for-long-tailed) 

(MoE) [Improving Long-Tailed Classification from Instance Level](https://arxiv.org/pdf/2104.06094.pdf) (preprint) [waiting code](https://paperswithcode.com/paper/improving-long-tailed-classification-from) 

> Instance level re-weighting.

(TADE) [Test-Agnostic Long-Tailed Recognition by Test-Time Aggregating Diverse Experts with Self-Supervision](https://arxiv.org/pdf/2107.09249.pdf) (preprint) [Code](https://github.com/Vanint/TADE-AgnosticLT) 

> Based on RIDE.



