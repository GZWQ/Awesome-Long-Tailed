



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
|   DRO-LT (ICCV 2021)    | 64.7 |  50.0  | 23.8 | 47.3  |
|  MetaSAug (CVPR 2021)   |  -   |   -    |  -   | 48.01 |
|    RIDE (ICLR 2021)     | 69.3 |  49.3  | 26.0 | 49.1  |
|     ACE (ICCV 2021)     | 66.3 |  52.8  | 27.2 | 49.6  |
|       TADE (2021)       |      |        |      | 49.8  |
|    BLMS (CVPR 2020)     |      |        |      | 50.8  |
|    PaCo (ICCV 2021)     |  -   |   -    |  -   | 52.0  |



## Places365_LT

Using ResNet-152 as backbone.

|                      | Many | Medium | Few  | All  |
| :------------------: | :--: | :----: | :--: | :--: |
|   LADE (CVPR 2021)   | 42.8 |  39.1  | 29.6 | 38.8 |
|    LDA (MM 2021)     | 41.0 |  40.7  | 32.1 | 39.1 |
|   RSG (CVPR 2021)    | 41.9 |  41.4  | 32.0 | 39.3 |
| DisAlign (CVPR 2021) | 40.4 |  42.4  | 30.1 | 39.3 |
| GistNet (ICCV 2021)  | 42.5 |  40.8  | 32.1 | 39.6 |
|  MiSLAS (CVPR 2021)  |      |        |      | 40.4 |
|     TADE (2021)      |      |        |      | 40.9 |
|   PaCo (ICCV 2021)   | 36.1 |  47.9  | 35.3 | 41.2 |



## ImageNet_LT

Using ResNet-50 as backbone (90 epochs)

|                         | Many | Medium | Few  |  All  |
| :---------------------: | :--: | :----: | :--: | :---: |
|  MetaSAug (CVPR 2021)   |      |        |      | 47.39 |
| LogitAdjust (ICLR 2021) |      |        |      | 51.1  |
|     SSP (NIPS2020)      | 63.2 |  48.1  | 29.2 | 51.3  |
|     KCL (ICLR 2021)     | 61.8 |  49.4  | 30.9 | 51.5  |
|   MiSLAS (CVPR 2021)    |  -   |   -    |  -   | 52.7  |
|  DisAlign (CVPR 2021)   | 61.3 |  52.2  | 31.4 | 52.9  |
|      LDA (MM 2021)      | 64.5 |  50.9  | 31.5 | 53.4  |
|   DRO-LT (ICCV 2021)    | 64.0 |  49.8  | 33.1 | 53.5  |
|     ACE (ICCV 2021)     |  -   |   -    |  -   | 54.7  |
|    RIDE (ICLR 2021)     |  -   |   -    |  -   | 55.4  |
|       CBD (2021)        | 68.5 |  52.7  | 29.2 | 55.6  |
|     SSD (ICCV 2021)     | 66.8 |  53.1  | 35.4 | 56.0  |
|       MoE (2021)        | 66.7 |  54.1  | 37.6 | 56.7  |
|    PaCo (ICCV 2021)     |  -   |   -    |  -   | 57.0  |
|       TADE (2021)       | 66.5 |  57.0  | 43.4 | 58.8  |



## iNaturalist18

Using ResNet-50 as backbone (90 epochs)

|                         | Many | Medium | Few  |  All  |
| :---------------------: | :--: | :----: | :--: | :---: |
| LogitAdjust (ICLR 2021) |  -   |   -    |  -   | 66.36 |
|  MetaSAug (CVPR 2021)   |  -   |   -    |  -   | 68.75 |
|     KCL (ICLR 2021)     |  -   |   -    |  -   | 68.6  |
|  DisAlign (CVPR 2021)   |      |        |      |       |
|     BBN (CVPR 2020)     |  -   |   -    |  -   | 69.62 |
|   DRO-LT (ICCV 2021)    |  -   |   -    |  -   | 69.7  |
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

[A Survey on Long-tailed Visual Recognition](https://arxiv.org/pdf/2205.13775.pdf) (IJCV 2022)

[Deep Long-Tailed Learning: A Survey](https://arxiv.org/pdf/2110.04596.pdf)  

[Bag of Tricks for Long-Tailed Visual Recognition with Deep Convolutional Neural Networks](http://www.lamda.nju.edu.cn/zhangys/papers/AAAI_tricks.pdf) (AAAI 2021) [Code](https://github.com/zhangyongshun/BagofTricks-LT)  



## Re-weighting

[Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification](https://arxiv.org/pdf/2112.14380.pdf) (AAAI 2022) [Code](https://github.com/BeierZhu/xERM) 

[A Re-Balancing Strategy for Class-Imbalanced Classification Based on Instance Difficulty](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_A_Re-Balancing_Strategy_for_Class-Imbalanced_Classification_Based_on_Instance_Difficulty_CVPR_2022_paper.pdf) (CVPR 2022)

[Long-Tailed Recognition via Weight Balancing](https://arxiv.org/pdf/2203.14197.pdf) (CVPR 2022) [Code](https://github.com/ShadeAlsha/LTR-weight-balancing)

[SOLVING THE LONG-TAILED PROBLEM VIA INTRA- AND INTER-CATEGORY BALANCE](https://arxiv.org/pdf/2204.09234.pdf) (ICASSP2022) 

[Influence-Balanced Loss for Imbalanced Visual Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Influence-Balanced_Loss_for_Imbalanced_Visual_Classification_ICCV_2021_paper.pdf) (ICCV 2021) [Code](https://github.com/pseulki/IB-Loss) 

(LADE) [Disentangling Label Distribution for Long-tailed Visual Recognition](https://arxiv.org/pdf/2012.00321.pdf) (CVPR 2021) [Code](https://github.com/hyperconnect/LADE) 

> An extension of BalancedSoftmax while improvement is quite limited. But it can handle unknown distributions of testing datasets.

(LogitAdjust) [Long-tail learning via logit adjustment](https://arxiv.org/pdf/2007.07314v1.pdf) (ICLR 2021)

(BLMS) [Balanced Meta-Softmax for Long-Tailed Visual Recognition](https://papers.nips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf) (NIPS 2020) [code](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification) 

[GradTail: Learning Long-Tailed Data Using Gradient-based Sample Weighting](https://arxiv.org/pdf/2201.05938.pdf) (preprint)

[Class-Aware Universum Inspired Re-Balance Learning for Long-Tailed Recognition](https://arxiv.org/pdf/2207.12808.pdf) (preprint)

[Class-Difficulty Based Methods for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2207.14499.pdf) (preprint)

[Margin Calibration for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2112.07225.pdf) (preprint)

## Sampling

[The Majority Can Help The Minority: Context-rich Minority Oversampling for Long-tailed Classification]() (CVPR 2022) [Code](https://github.com/naver-ai/cmo) 

[Solving Long-tailed Recognition with Deep Realistic Taxonomic Classifier](https://arxiv.org/pdf/2007.09898.pdf) (ECCV 2020) [Code](https://github.com/gina9726/Deep-RTC) 

## Meta Learning

(MetaSAug) [MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2103.12579.pdf) (CVPR 2021) [code](https://github.com/BIT-DA/MetaSAug) 

[Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition from a Domain Adaptation Perspective](https://arxiv.org/pdf/2003.10780.pdf) (CVPR 2020) [code](https://github.com/abdullahjamal/Longtail_DA) 

[Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://arxiv.org/pdf/1902.07379.pdf) (NIPS 2019) [code](https://github.com/xjtushujun/Meta-weight-net_class-imbalance)  



## Feature Manipulation

[Constructing Balance from Imbalance for Long-tailed Image Recognition](https://arxiv.org/pdf/2208.02567.pdf) (ECCV 2022) [Code](https://github.com/silicx/DLSA) 

[Invariant Feature Learning for Generalized Long-Tailed Classification](https://arxiv.org/pdf/2207.09504.pdf) (ECCV 2022) [Code](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch) 

[VL-LTR: Learning Class-wise Visual-Linguistic Representation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2111.13579.pdf) (ECCV 2022) [Code](https://github.com/ChangyaoTian/VL-LTR) 

[Balanced Contrastive Learning for Long-Tailed Visual Recognition](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.pdf) (CVPR 2022) [Code](https://github.com/FlamieZhu/Balanced-Contrastive-Learning) 

[Retrieval Augmented Classification for Long-Tail Visual Recognition](https://arxiv.org/pdf/2202.11233.pdf) (CVPR 2022)

[Long-tail Recognition via Compositional Knowledge Transfer](https://arxiv.org/pdf/2112.06741.pdf) (CVPR 2022)

[Targeted Supervised Contrastive Learning for Long-Tailed Recognition]() (CVPR 2022) [Code](https://github.com/LTH14/targeted-supcon) 

[Self-supervised Learning is More Robust to Dataset Imbalance](https://openreview.net/forum?id=4AZz9osqrar) (ICLR 2022)

[Imagine by Reasoning: A Reasoning-Based Implicit Semantic Data Augmentation for Long-Tailed Classification](https://arxiv.org/pdf/2112.07928.pdf) (AAAI 2022) [Code](https://github.com/xiaohua-chen/risda) 

[Do deep networks transfer invariances across classes?](https://openreview.net/forum?id=Fn7i_r5rR0q) (ICLR 2022)

> Experiments on only small-sacle datasets.

(SSD) [Self Supervision to Distillation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2109.04075.pdf) [ICCV 2021] [waiting code](https://github.com/MCG-NJU/SSD-LT) 

[Procrustean Training for Imbalanced Deep Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Procrustean_Training_for_Imbalanced_Deep_Learning_ICCV_2021_paper.pdf) (ICCV 2021) 

[Distributional Robustness Loss for Long-tail Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Samuel_Distributional_Robustness_Loss_for_Long-Tail_Learning_ICCV_2021_paper.pdf) (ICCV 2021) [Code](https://github.com/dvirsamuel/DRO-LT) 

> This method focuses on improving the learned representation at the penultimate layer by using an extra robustness loss. The robustness loss is computed in the feature level and kind of like contrastive loss.

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

[Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective](https://arxiv.org/pdf/2002.10826.pdf) (CVPR 2020)

[Feature Transfer Learning for Deep Face Recognition with Under-Represented Data](https://arxiv.org/pdf/1803.09014.pdf) (CVPR 2019)

[Memory-based Jitter: Improving Visual Recognition on Long-tailed Data with Diversity In Memory](https://arxiv.org/pdf/2008.09809.pdf) (preprint)

(BKD) [Balanced Knowledge Distillation for Long-tailed Learning](https://arxiv.org/pdf/2104.10510.pdf) preprint [Code](https://github.com/EricZsy/BalancedKnowledgeDistillation) 

[A Simple Long-Tailed Recognition Baseline via Vision-Language Model](https://arxiv.org/pdf/2111.14745.pdf) (preprint) [Code](https://github.com/gaopengcuhk/BALLAD) 

[Feature Generation for Long-tail Classification](https://arxiv.org/pdf/2111.05956.pdf) [Code](https://github.com/rahulvigneswaran/TailCalibX) (preprint)

[Long-Tailed Classification with Gradual Balanced Loss and Adaptive Feature Generation](https://arxiv.org/pdf/2203.00452.pdf) (preprint)

[Rebalanced Siamese Contrastive Mining for Long-Tailed Recognition](https://arxiv.org/pdf/2203.11506.pdf) (preprint) [Code](https://github.com/dvlab-research/Imbalanced-Learning) 

### Logit Adjustment

[Label-Aware Distribution Calibration for Long-tailed Classification](https://arxiv.org/pdf/2111.04901.pdf) (AAAI 2022)

[Long-Tailed Recognition via Weight Balancing](file:///Users/daniel/Desktop/Long-Tailed%20Recognition%20via%20Weight%20Balancing.pdf) (CVPR 2022) [Code](https://github.com/ShadeAlsha/LTR-weight-balancing)

[Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment]() (CVPR 2022)

[Optimal Transport for Long-Tailed Recognition with Learnable Cost Matrix](https://openreview.net/forum?id=t98k9ePQQpn) (ICLR 2022)

[Towards Calibrated Model for Long-Tailed Visual Recognition from Prior Perspective](https://openreview.net/pdf?id=vqzAfN-BoA_) (NIPS2021) [Code](https://github.com/XuZhengzhuo/Prior-LT) 

[Normalization Calibration (NorCal) for Long-Tailed Object Detection and Instance Segmentation](https://openreview.net/pdf?id=t9gKUW9T8fX)  (NIPS2021) [Code](https://github.com/tydpan/NorCal)  

[t-vMF Similarity for Regularizing Intra-Class Feature Distribution](https://staff.aist.go.jp/takumi.kobayashi/publication/2021/CVPR2021.pdf) (CVPR2021) [Code](https://github.com/tk1980/tvMF) 

[Distribution Alignment: A Unified Framework for Long-tail Visual Recognition](https://arxiv.org/pdf/2103.16370.pdf) (CVPR 2021) [Code](https://github.com/Megvii-BaseDetection/DisAlign) 

(DisAlign) [Distilling Virtual Examples for Long-tailed Recognition](https://arxiv.org/pdf/2103.15042.pdf) (CVPR 2021)

> Two-stage decoupling based. Calibrate the classifier and align the model prediction with the desired distribution favoring the balanced predcition. 

(LDA) [Long-tailed Distribution Adaptation](https://arxiv.org/pdf/2110.02686.pdf) (ACM MM 2021) [Code](https://github.com/pengzhiliang/LDA) 

[Long-Tail Learning via Logit Adjustment](https://arxiv.org/pdf/2007.07314.pdf) (ICLR 2021) [Code](https://github.com/google-research/google-research/tree/master/logit_adjustment) 

[ELM: Embedding and Logit Margins for Long-Tail Learning](https://arxiv.org/pdf/2204.13208.pdf) (preprint) 

## Multiple Experts

[Trustworthy Long-Tailed Classification](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Trustworthy_Long-Tailed_Classification_CVPR_2022_paper.pdf) (CVPR 2022) [Code](https://github.com/lblaoke/TLC) 

[Nested Collaborative Learning for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2203.15359.pdf) (CVPR 2022) [Code](https://github.com/Bazinga699/NCL) 

(ACE) [ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot](https://arxiv.org/pdf/2108.02385.pdf) (ICCV 2021) [Code](https://github.com/jrcai/ACE) 

(RIDE) [Long-tailed Recognition by Routing Diverse Distribution-Aware Experts](https://arxiv.org/pdf/2010.01809.pdf) (ICLR 2021) [Code](https://github.com/frank-xwang/RIDE-LongTailRecognition) 

(CBD) [Class-Balanced Distillation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2104.05279.pdf) (BMVC 2021) [Code](https://github.com/google-research/google-research/tree/master/class_balanced_distillation)  

> A two-stage solution. The first stage is to learn multiple teacher models distinguished by different data augmentation strategies. In Stage2, a student model is trained from scratch distilled by teachers' knowledge in the feature space instead of the classifier layer.

[Long-Tailed Recognition Using Class-Balanced Experts](https://arxiv.org/pdf/2004.03706.pdf) (DAGM-GCPR 2020) [Code](https://github.com/ssfootball04/class-balanced-experts) 

(MoE) [Improving Long-Tailed Classification from Instance Level](https://arxiv.org/pdf/2104.06094.pdf) (preprint) [waiting code](https://paperswithcode.com/paper/improving-long-tailed-classification-from) 

> Instance level re-weighting.

(TADE) [Test-Agnostic Long-Tailed Recognition by Test-Time Aggregating Diverse Experts with Self-Supervision](https://arxiv.org/pdf/2107.09249.pdf) (preprint) [Code](https://github.com/Vanint/TADE-AgnosticLT) 

> Based on RIDE. 

[Balanced Product of Experts for Long-Tailed Recognition](https://arxiv.org/pdf/2206.05260.pdf) (preprint)

[DBN-Mix: Training Dual Branch Network Using Bilateral Mixup Augmentation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2207.02173.pdf) (preprint)

# New Task Settings

|        Method        | AP_b  | AP_f  | AP_c  | AP_r  |  AP   |
| :------------------: | :---: | :---: | :---: | :---: | :---: |
|  SoftmaxCE(Uniform)  |   -   | 29.3  | 17.4  |  1.2  | 19.3  |
|    SoftmaxCE(RFS)    |   -   | 28.3  | 21.6  | 12.9  | 22.8  |
| MosaicOS (ICCV 2021) | 25.05 | 28.83 | 22.99 | 18.17 | 24.45 |
|  EQL v2 (CVPR 2021)  | 26.1  | 30.2  | 24.3  | 17.7  | 25.5  |
|   ACSL (CVPR 2021)   |   -   | 29.37 | 26.41 | 18.64 | 26.36 |
|  Seesaw (CVPR 2021)  | 27.4  | 29.8  | 26.1  | 19.6  | 26.4  |
|   LOCE (ICCV 2021)   | 27.4  | 30.7  | 26.2  | 18.5  | 26.6  |
|  NORCAL (NIPS 2021)  | 27.77 | 29.10 | 25.82 | 23.86 | 26.76 |
|   FASA (ICCV 2021)   |   -   | 30.1  | 27.5  | 21.0  | 27.5  |

> - Results on LVIS-v1.0.
> - ResNet50-FPN is the backbone. 
> - The evaluation metric is AP across IoU threshold from 0.5 to 0.95 over all categories. [Link](https://arxiv.org/pdf/2003.05176.pdf) 
> - AP_b denotes the detection performance while AP denotes the segmentation results.

[Long-tailed Instance Segmentation using Gumbel Optimized Loss](https://arxiv.org/pdf/2207.10936.pdf) (ECCV 2022) [Code](https://github.com/kostas1515/GOL) 

[Equalized Focal Loss for Dense Long-Tailed Object Detection](https://arxiv.org/pdf/2201.02593.pdf) (CVPR 2022) [Code](https://github.com/ModelTC/EOD) 

[Relieving Long-tailed Instance Segmentation via Pairwise Class Balance](https://arxiv.org/pdf/2201.02784.pdf) (CVPR 2022) [Code](https://github.com/megvii-research/PCB) 

[Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details](https://arxiv.org/pdf/2102.01066.pdf) (Preprint) [Code](https://github.com/achalddave/large-vocab-devil) 

> 

(NORCAL) [On Model Calibration for Long-Tailed Object Detection and Instance Segmentation](https://arxiv.org/pdf/2107.02170.pdf) (NIPS 2021) [Code](https://github.com/tydpan/NorCal) 

> Post-processing of logits. Except the background logit, a temperature-like hyperparameter is used to scale down the logits, which is based on the class frequence. I also applied this post-processing techinique on Long-tailed classification and only effective in CIFAR10-LT and CIFAR100-LT but not in large scale dataset like ImageNet-LT.

(FASA) [FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation](https://arxiv.org/pdf/2102.12867.pdf)  (ICCV 2021) [Code](https://github.com/yuhangzang/FASA)   

> 1. Augmentation is performed in the feature space.
> 2. For a class to be augmentated, we can compute its mean features from current batch data and then update it in a momentum way. So we can generate virtual features by firstly randomly sampling variation from a gaussian prior and add the variation to the mean feature.

(LOCE) [Exploring Classification Equilibrium in Long-Tailed Object Detection](https://arxiv.org/pdf/2108.07507.pdf) (ICCV 2021) [Code](https://github.com/fcjian/LOCE) 

> Two type of strategies are used: logit margin adjustment and object-level features augmentation. Both strategies are based on the mean class accuraies instead of the class frequency. 

(MosaicOS) [MOSAICOS: A Simple and Effective Use of Object-Centric Images for Long-Tailed Object Detection](https://arxiv.org/pdf/2102.08884.pdf) (ICCV 2021) [Code](https://github.com/czhang0528/MosaicOS/) 

> Data augmentation method. Utilize object images from ImageNet to create pseudo scene image, and fine-tune the model.

[Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/pdf/2012.07177.pdf) (CVPR 2021) [Code](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/copy_paste) 

> Effective in both COCO and LVIS.

(ACSL) [Adaptive Class Suppression Loss for Long-Tail Object Detection](https://arxiv.org/pdf/2104.00885.pdf) (CVPR 2021) [Code](https://github.com/CASIA-IVA-Lab/ACSL) 

> Similar to Equalization Loss and Balanced Group Softmax, it also tries to supress the negative graidents on tail classes. But instead of using prior frequency information as the signal, it uses the prediction confidence to decide whether suppresses the negative gradients.

(Seesaw) [Seesaw Loss for Long-Tailed Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Seesaw_Loss_for_Long-Tailed_Instance_Segmentation_CVPR_2021_paper.pdf) (CVPR 2021) [Code](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss) 

> Cross entropy adjusting based. Two scaling factor is applied in the softmax. The first is to decrease the graident scale on tail classes when the input image belongs to a head class. The second is increse the gradient panelty on any classes if it is predicted wrongly.

(EQL v2) [Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Tan_Equalization_Loss_v2_A_New_Gradient_Balance_Approach_for_Long-Tailed_CVPR_2021_paper.pdf) (CVPR 2021) [Code](https://github.com/tztztztztz/eqlv2) 

> Gradient-guided re-weighting

[DropLoss for Long-Tail Instance Segmentation](https://arxiv.org/pdf/2104.06402.pdf) (AAAI 2021) [Code](https://github.com/timy90022/DropLoss) 

> Unlike EQL, the proposed solution tries to remove the supression from background classes.

[Image-Level or Object-Level? A Tale of Two Resampling Strategies for Long-Tailed Detection](https://arxiv.org/pdf/2104.05702.pdf) (ICML 2021) [Code](https://github.com/NVlabs/RIO) 

> Bi-level sampling. Repeat factor sampling (RFS) is used for image-level sampling. As for the object-level sampling, they propose a memory bank to store (feat, bbx) of tail classes so that more tail objects can be sampled during each batch.

[Learning to Segment the Tail](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Learning_to_Segment_the_Tail_CVPR_2020_paper.pdf) (CVPR 2020) [Code](https://github.com/JoyHuYY1412/LST_LVIS) 

> 

[Equalization Loss for Long-Tailed Object Recognition](https://arxiv.org/pdf/2003.05176.pdf) (CVPR 2020) [Code](https://github.com/tztztztztz/eql.detectron2) 

> Ignore all gradients from head classes when updating tail classifiers.

[Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax](https://arxiv.org/pdf/2006.10408.pdf) (CVPR 2020) [Code](https://github.com/FishYuLi/BalancedGroupSoftmax) 

> Categories with similar numbers of training instances into the same group and computes group-wise softmax crossentropy loss respectively.

[Imbalanced Continual Learning with Partitioning Reservoir Sampling](https://arxiv.org/pdf/2009.03632.pdf) (ECCV 2020) [Code](https://github.com/cdjkim/PRS) 

[The Devil is in Classification: A Simple Framework for Long-tail Instance Segmentation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590715.pdf) (ECCV 2020) [Code](https://github.com/twangnh/SimCal) 

> Retraining the classification head only with a bi-level sampling scheme. During the inference stage, two classifiers are combined to make the final prediction. 

[Forest R-CNN: Large-Vocabulary Long-Tailed Object Detection and Instance Segmentation](https://arxiv.org/pdf/2008.05676.pdf) (ACM MM 2020) [Code](https://github.com/JialianW/Forest_RCNN)  



[Equalized Focal Loss for Dense Long-Tailed Object Detection](https://arxiv.org/pdf/2201.02593.pdf) (CVPR 2022) [Code](https://github.com/ModelTC/United-Perception) 

[On Multi-Domain Long-Tailed Recognition, Imbalanced Domain Generalization and Beyond](https://arxiv.org/pdf/2203.09513.pdf) (ECCV 2022) [Code](https://github.com/YyzHarry/multi-domain-imbalance) 

[Tackling Long-Tailed Category Distribution Under Domain Shifts](https://arxiv.org/pdf/2207.10150.pdf) (ECCV 2022) [Code](https://github.com/guxiao0822/lt-ds)

[Identifying Hard Noise in Long-Tailed Sample Distribution](https://arxiv.org/pdf/2207.13378.pdf) (ECCV 2022) [Code](https://github.com/yxymessi/H2E-Framework) 

[VideoLT: Large-scale Long-tailed Video Recognition](https://arxiv.org/pdf/2105.02668.pdf) (ICCV 2021) [Code](https://github.com/17Skye17/VideoLT) 

[Adversarial Robustness under Long-Tailed Distribution](https://arxiv.org/pdf/2104.02703.pdf) (CVPR 2021) [Code](https://github.com/wutong16/Adversarial_Long-Tail) 

[Calibrating Concepts and Operations: Towards Symbolic Reasoning on Real Images]()

[BGT-Net: Bidirectional GRU Transformer Network for Scene Graph Generation]()

[Google Landmarks Dataset v2 -- A Large-Scale Benchmark for Instance-Level Recognition and Retrieval]

[Learning of Visual Relations: The Devil is in the Tails]

[CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wei_CReST_A_Class-Rebalancing_Self-Training_Framework_for_Imbalanced_Semi-Supervised_Learning_CVPR_2021_paper.pdf) (CVPR 2021) [Code](https://github.com/google-research/crest) 

[Unsupervised Discovery of the Long-Tail in Instance Segmentation Using Hierarchical Self-Supervision](https://openaccess.thecvf.com/content/CVPR2021/papers/Weng_Unsupervised_Discovery_of_the_Long-Tail_in_Instance_Segmentation_Using_Hierarchical_CVPR_2021_paper.pdf) (CVPR 2021) [Code](https://github.com/ZZWENG/longtail_segmentation)   
