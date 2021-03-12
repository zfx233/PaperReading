# PaperReading

|    Survey Paper    |  Conference  | Abstract |
|  :---------  | :------:  | :------: |
|  [A Survey of Clustering With Deep Learning: From the Perspective of Network Architecture](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8412085)  |  IEEE ACCESS 2018  |从网络体系结构的角度对深度聚类进行了研究。首先介绍了一些初步的知识（前馈全连接神经网络、前向卷积神经网络、深度信任网络、自编码、GAN & VAE）。然后分别基于AE、CDNN、VAE和GAN等网络体系结构进行了研究，并介绍了一些具有代表性的方法。 |


|   Method  Paper    |  Method |  Conference |  Code | Result(ACC) | Abstract |
|  :---------  | :------:  | :------: | :------: | :------: | :------: |
|  [Unsupervised Deep Embedding for Clustering Analysis](http://proceedings.mlr.press/v48/xieb16.pdf) |  DEC  |   ICML 2016  | [Caffe](https://github.com/piiswrong/dec) [TensorFlow](https://github.com/danathughes/DeepEmbeddedClustering) [Keras](https://github.com/XifengGuo/DEC-keras) | MNIST: 0.843 | 最具代表性的深度聚类方法之一。预训练AE的参数，使用encoder作为网络架构。通过最小化soft_ass和target_distrbution之间的KL散度迭代优化参数。其本质是使soft_ass中概率极端化，大的越大，小的越小，所以受pretrain效果的影响较大。 |
| [Improved Deep Embedded Clustering with Local Structure Preservation](https://www.ijcai.org/proceedings/2017/0243.pdf)   |  IDEC  |   IJCAI 2017  |  [Keras](https://github.com/XifengGuo/IDEC) [Pytorch](https://github.com/dawnranger/IDEC-pytorch) | MNIST: 0.881 | 主要优化是在DEC的基础上添加了一个reconstruction loss，为训练参数设定了一个约束，使得训练过程中尽量不破坏嵌入，同时带来一些波动，从而获得更好的结果。本文还提出了一个trick：设置固定 p 的轮数，使 p 相对固定，算法更稳定。实验中发现这个trick效果相当不错。 |
| [Deep Adaptive Image Clustering](https://openaccess.thecvf.com/content_ICCV_2017/papers/Chang_Deep_Adaptive_Image_ICCV_2017_paper.pdf)   | DAC | ICCV 2017 | [Keras](https://github.com/vector-1127/DAC) | MNIST: 0.978  CIFAR10: 0.522 |提出了一种端到端的算法，基于二分类的思想，通过预测两个sample之间是否属于同一类来训练。网络输出为soft assignment，计算相似度矩阵并构造目标矩阵（由01组成），根据adaptive的阈值选取sample来训练（即计算loss时的weight矩阵，由01组成）。训练网络和更新阈值两者交替迭代。 |
| [Joint Unsupervised Learning of Deep Representations and Image Clustering](https://arxiv.org/pdf/1604.03628.pdf) |  JULE  | CVPR 2016 |   [Torch](https://github.com/jwyang/JULE.torch) |  |  |
|  |  |  |  |  |  |
| [Adversarial Graph Embedding for Ensemble Clustering](https://www.ijcai.org/proceedings/2019/0494.pdf) |  AGAE  |  IJCAI 2019  |  | - | 根据相似性从数据中发掘并构建图结构，采用GCN作为encoder，联合特征和共识图信息，生成嵌入表示，以内积层作为decoder，计算图重构loss。另一方面，根据嵌入表示和簇中心计算软分布，构造新的嵌入表示，作为正样本，原有的嵌入表示作为负样本，使用对抗学习的思想进行训练。  |
| [Unsupervised Clustering using Pseudo-semi-supervised Learning](https://openreview.net/attachment?id=rJlnxkSYPS&name=original_pdf) | Kingdra | ICLR 2020 |  | MNIST: 0.985 CIFAR10: 0.546 | 使用了半监督学习的思想，训练的label使用构造得到高置信度label。首先通过多个模型来获取样本两两间的关系，若大多数模型都认为两个样本属于同一类或不同类，则将对应的关系添加到构造的graph中：样本作为点，边分为正边和负边，代表了属于同一类和不属于同一类。从图上提取k个置信度高的簇，分配label，作为自监督学习中的labelde小数据集训练模型，迭代进行。 |
| [Contrastive Clustering](https://arxiv.org/pdf/2009.09687.pdf) | CC | AAAI 2021 | [PyTorch](https://github.com/Yunfan-Li/Contrastive-Clustering) | CIFAR10: 0.790 | 一种端到端算法，主要思想是样本层面和簇层面的对比学习。通过数据增强构造正样本对和负样本对。特征矩阵的行可以看做样本的软分配，而列可以看做簇的嵌入表示，以此分别在行空间和列空间做对比学习，同时优化两者的loss。 |
| [SCAN: Learning to Classify Images without Labels](https://arxiv.org/abs/2005.12320) | SCAN | ECCV2020 | [Pytrorch](https://github.com/wvangansbeke/Unsupervised-Classification) | CIFAR10: 0.883 |  two-step: 表征学习和聚类。  表征学习阶段，选择一种pretext任务(SimCLR)，通过自监督学习的方式来学习样本嵌入。 之后，为每个样本选取K个相邻样本。 聚类阶段，输出为样本的软分配。首先通过SCAN-loss优化网络，接着迭代选取高置信度样本，分配伪标签，通过cross-entropy loss训练网络。SCAN-loss分两部分，第一部分使得样本和它的邻居样本之间的距离最小，第二部分为熵约束。 |
|[Improving Unsupervised Image Clustering With Robust Learning](https://arxiv.org/pdf/2012.11150.pdf)|RUC|Arxiv 2020|[Pytorch](https://github.com/deu30303/RUC)|  |  |
|[MiCE: Mixture of Contrastive Experts for Unsupervised Image Clustering ](https://openreview.net/pdf?id=gV3wdEOGy_V)|MiCE|ICLR2021|  |  |



