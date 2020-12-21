# PaperReading

|    Survey Paper    |  Conference  | Abstract |
|  :---------  | :------:  | :------: |
|  [A Survey of Clustering With Deep Learning: From the Perspective of Network Architecture](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8412085)  |  IEEE ACCESS 2018  |从网络体系结构的角度对深度聚类进行了研究。首先介绍了一些初步的知识（前馈全连接神经网络、前向卷积神经网络、深度信任网络、自编码、GAN & VAE）。然后分别基于AE、CDNN、VAE和GAN等网络体系结构进行了研究，并介绍了一些具有代表性的方法。 |


|   Component Paper   | Conference  |  Code | Abstract |
|  :---------  | :------:  | :------: | :------: |
| [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://jmlr.csail.mit.edu/papers/volume11/vincent10a/vincent10a.pdf) | JMLR 2010 |  |


|   Method  Paper    |  Method |  Conference |  Code | Abstract |
|  :---------  | :------:  | :------: | :------: | :------: |
|  [Unsupervised Deep Embedding for Clustering Analysis](http://proceedings.mlr.press/v48/xieb16.pdf) |  DEC  |   ICML 2016  | [Caffe](https://github.com/piiswrong/dec) [TensorFlow](https://github.com/danathughes/DeepEmbeddedClustering) [Keras](https://github.com/XifengGuo/DEC-keras) | 最具代表性的深度聚类方法之一。预训练AE的参数，使用encoder作为网络架构。通过最小化soft_ass和target_distrbution之间的KL散度迭代优化参数。其本质是使soft_ass中概率极端化，大的越大，小的越小，所以受pretrain效果的影响较大。 |
| [Deep Semantic Clustering by Partition Confidence Maximisation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Deep_Semantic_Clustering_by_Partition_Confidence_Maximisation_CVPR_2020_paper.pdf) | PICA | CVPR 2020 |  |
| [Improved Deep Embedded Clustering with Local Structure Preservation](https://www.ijcai.org/proceedings/2017/0243.pdf)   |  IDEC  |   IJCAI 2017  |  [Keras](https://github.com/XifengGuo/IDEC),[Pytorch](https://github.com/dawnranger/IDEC-pytorch) | 主要优化是在DEC的基础上添加了一个reconstruction loss，为训练参数设定了一个约束，使得训练过程中尽量不破坏嵌入，同时带来一些波动，从而获得更好的结果。本文还提出了一个trick：设置固定 p 的轮数，使 p 相对固定，算法更稳定。实验中发现这个trick效果相当不错。 |
