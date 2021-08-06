# PaperReading

|    Survey Paper    |  Conference  | Abstract |
|  :---------  | :------:  | :------: |
|  [A Survey of Clustering With Deep Learning: From the Perspective of Network Architecture](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8412085)  |  IEEE ACCESS 2018  |从网络体系结构的角度对深度聚类进行了研究。首先介绍了一些初步的知识（前馈全连接神经网络、前向卷积神经网络、深度信任网络、自编码、GAN & VAE）。然后分别基于AE、CDNN、VAE和GAN等网络体系结构进行了研究，并介绍了一些具有代表性的方法。 |

<br><br>

|   Method  Paper    |  Method |  Conference |  Code | Result(ACC) | Abstract |
|  :---------  | :------:  | :------: | :------: | :------: | :------: |
|  [Unsupervised Deep Embedding for Clustering Analysis](http://proceedings.mlr.press/v48/xieb16.pdf) |  DEC  |   ICML 2016  | [Caffe](https://github.com/piiswrong/dec) [TensorFlow](https://github.com/danathughes/DeepEmbeddedClustering) [Keras](https://github.com/XifengGuo/DEC-keras) | MNIST: 0.843 | 最具代表性的深度聚类方法之一。预训练AE的参数，使用encoder作为网络架构。通过最小化soft_ass和target_distrbution之间的KL散度迭代优化参数。其本质是使soft_ass中概率极端化，大的越大，小的越小，所以受pretrain效果的影响较大。 |
| [Improved Deep Embedded Clustering with Local Structure Preservation](https://www.ijcai.org/proceedings/2017/0243.pdf)   |  IDEC  |   IJCAI 2017  |  [Keras](https://github.com/XifengGuo/IDEC) [Pytorch](https://github.com/dawnranger/IDEC-pytorch) | MNIST: 0.881 | 主要优化是在DEC的基础上添加了一个reconstruction loss，为训练参数设定了一个约束，使得训练过程中尽量不破坏嵌入，同时带来一些波动，从而获得更好的结果。本文还提出了一个trick：设置固定 p 的轮数，使 p 相对固定，算法更稳定。实验中发现这个trick效果相当不错。 |
| [Deep Adaptive Image Clustering](https://openaccess.thecvf.com/content_ICCV_2017/papers/Chang_Deep_Adaptive_Image_ICCV_2017_paper.pdf)   | DAC | ICCV 2017 | [Keras](https://github.com/vector-1127/DAC) | MNIST: 0.978  CIFAR10: 0.522 |提出了一种端到端的算法，基于二分类的思想，通过预测两个sample之间是否属于同一类来训练。网络输出为soft assignment，计算相似度矩阵并构造目标矩阵（由01组成），根据adaptive的阈值选取sample来训练（即计算loss时的weight矩阵，由01组成）。训练网络和更新阈值两者交替迭代。 |
| [Associative Deep ClusteringTraining a Classification Network with no Labels](https://vision.cs.tum.edu/_media/spezial/bib/haeusser18associative.pdf) | ADC | GCPR 2018 | - | MNIST: 0.973 CIFAR10: 0.325 | 缝合怪。将Associative learning的思想用于聚类，大致思想是在labeled数据和unlabeled数据之间构建转换关系，以labeled数据的标签作为目标来训练。思路是好得多，但是它将许多loss组合成最终的loss，可解释性比较差，也非常难以复现，效果也不是很好，所以不是非常有意义。 |
| [Adversarial Graph Embedding for Ensemble Clustering](https://www.ijcai.org/proceedings/2019/0494.pdf) |  AGAE  |  IJCAI 2019  | - | - | 根据相似性从数据中发掘并构建图结构，采用GCN作为encoder，联合特征和共识图信息，生成嵌入表示，以内积层作为decoder，计算图重构loss。另一方面，根据嵌入表示和簇中心计算软分布，构造新的嵌入表示，作为正样本，原有的嵌入表示作为负样本，使用对抗学习的思想进行训练。  |
| [Unsupervised Clustering using Pseudo-semi-supervised Learning](https://openreview.net/attachment?id=rJlnxkSYPS&name=original_pdf) | Kingdra | ICLR 2020 | [Keras](https://github.com/divamgupta/deep-clustering-kingdra) | MNIST: 0.985 CIFAR10: 0.546 | 使用了半监督学习的思想，训练的label使用构造得到高置信度label。首先通过多个模型来获取样本两两间的关系，若大多数模型都认为两个样本属于同一类或不同类，则将对应的关系添加到构造的graph中：样本作为点，边分为正边和负边，代表了属于同一类和不属于同一类。从图上提取k个置信度高的簇，分配label，作为自监督学习中的labelde小数据集训练模型，迭代进行。 |
| [Contrastive Clustering](https://arxiv.org/pdf/2009.09687.pdf) | CC | AAAI 2021 | [PyTorch](https://github.com/Yunfan-Li/Contrastive-Clustering) | CIFAR10: 0.790 | 一种端到端算法，主要思想是样本层面和簇层面的对比学习。通过数据增强构造正样本对和负样本对。特征矩阵的行可以看做样本的软分配，而列可以看做簇的嵌入表示，以此分别在行空间和列空间做对比学习，同时优化两者的loss。 |
| [SCAN: Learning to Classify Images without Labels](https://arxiv.org/abs/2005.12320) | SCAN | ECCV 2020 | [Pytrorch](https://github.com/wvangansbeke/Unsupervised-Classification) | CIFAR10: 0.883 |  two-step: 表征学习和聚类。  表征学习阶段，选择一种pretext任务(SimCLR)，通过自监督学习的方式来学习样本嵌入。 之后，为每个样本选取K个相邻样本。 聚类阶段，输出为样本的软分配。首先通过SCAN-loss优化网络，接着迭代选取高置信度样本，分配伪标签，通过cross-entropy loss训练网络。SCAN-loss分两部分，第一部分使得样本和它的邻居样本之间的距离最小，第二部分为熵约束。 |
|[Improving Unsupervised Image Clustering With Robust Learning](https://arxiv.org/pdf/2012.11150.pdf)| RUC | CVPR 2021 |[Pytorch](https://github.com/deu30303/RUC)| CIFAR10: 0.903 | RUC是一个可叠加的模块，以其他方法的最终模型为输入，在此基础上使用一些 Robust Learning 的方法（Co-train、label smoothing）来提升模型效果。具体模型如下：首先根据原始模型输出的 pseudo label 将数据集划分为 labeled data 和 unlabeled data，接着同时训练两个模型，使用 Co-refine 生成更为可靠的label，应用 Mix-Match 的半监督学习框架，结合 smoothed-label data 计算最终的 loss，迭代地更新两个网络。每个 epoch 的最后更新 labeled data 和 unlabeled data 的划分。 |
|[MiCE: Mixture of Contrastive Experts for Unsupervised Image Clustering ](https://openreview.net/pdf?id=gV3wdEOGy_V)|MiCE|ICLR2021| [Pytorch](https://github.com/TsungWeiTsai/MiCE) | CIFAR10: 0.835 | 框架比较特殊。受MoE的启发，MiCE使用了分治的思想,通过K个experts来预测样本的分配。每个expert有自己的“知识”，对样本分配提出自己的预测，而一个样本对每个expert的信任程度也是不一样的。还有一个特殊的地方为：MiCE的主体网络将一张图片(1 x image_size)映射到 K x d 的空间中，再将每个 1 x d 的embedding分别输入到对应的expert中。实际上，如果将expert看做网络的一部分，MiCE其实也是使用了K个不同的网络来预测分配，不同的地方在于网络的前一部分的参数是共享的，且expert预测的分配也有权重。 |
| [SPICE: Semantic Pseudo-labeling for Image Clustering](https://arxiv.org/pdf/2103.09382v1.pdf) | SPICE | Arxiv 2021 | [Pytorch](https://github.com/niuchuangnn/SPICE) | CIFAR10: 0.917 | SPICE是一个三阶段的方法，有一些缝合怪的意思，但其组件和思想还是比较简单的。模型包括backbone(ResNet34)和一个分类器CLSHead(MLP)。阶段一训练表征学习模型backbone，模型的参数在后续任务中是frozen的。阶段二 SPICE-Self 自监督学习阶段，通过高置信度样本构造聚类中心，为中心周围的样本分配伪标签，训练分类器。阶段三 SPICE-Semi “半监督”学习阶段根据伪标签和语义相邻关系选取labeled样本，以半监督的方式训练分类器。 |
|[Clustering-friendly Representation Learning via Instance Discrimination and Feature Decorrelation](https://openreview.net/pdf?id=e12NDM7wkEY)|IDFD|ICLR2021| [Pytorch](https://github.com/TTN-YKK/Clustering_friendly_representation_learning) | CIFAR10: 0.828 | IDFD可以说是一个面向聚类任务的表征学习方法。它有两个目标：学习样本间的相似性和减少特征内的相关性。具体的思想和CC有一些相似，在一个n*d的特征矩阵中，行向量作为样本特征，列向量作为特征的表示。优化的目标都是使样本或特征与自己的相似度最大，与其他样本或特征得到相似度最小。IDFD的聚类方法使用了最简单的K-Means，但是效果还是比较好的，说明IDFD在表征学习上做得非常不错。IDFD整体的框架非常简单，如果配合好一点的聚类方法，效果应该还能再提升一些。 |
| [Barlow Twins：Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf) | BT | Arxiv 2021 | [Pytorch](https://github.com/facebookresearch/barlowtwins) | - | BT是一个表征学习方法，框架非常简单：样本X的两个增强样本Ya和Yb通过相同的网络，得到表征Za和Zb。但它有一点比较特殊：它求的是两个增强样本之间的相似度矩阵，即 d*d 的矩阵。目标函数也非常简单，分两项，第一项使得对角线元素尽量接近，第二项使得非对角线元素尽量接近0。由于比较的是同一个样本的两个增强之间的关系，所以BT对 batch_size 没有要求，鲁棒性较强。相应的，特征维度的增加非常有利于BT的性能。 |
| [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/pdf/2006.09882.pdf) | SwAV | NIPS2020 |[PyTorch](https://github.com/facebookresearch/swav)| - | SwAV 的框架比较简单：sample x 的多个数据增强通过网络，获得 embedding z，根据 z 和聚类中心 c 计算 label q，再通过最小化不同增强之间的差异来训练网络。创新点：数据增强使用了 mutil-crop，除了两个标准 size 的增强，mutil-crop 还使用了 V 个较小 size 的增强，使用两个标准增强的 label 来训练 V 个小 size 增强的预测，目标是最小化两者之间的交叉熵。 |
|[Mitigating Embedding and Class Assignment Mismatch in Unsupervised Image Classification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690749.pdf)|TSUC|ECCV 2020|[Pytorch](https://github.com/dscig/TwoStageUC)| CIFAR10: 0.810 | TSUC 是一个 two-stage 的方法。Stage 1 使用了 Super-AND(一种样本特异性学习方法)来进行预训练，以获得良好的网络初始化。Stage 2 先 fineturn embedding，再通过5个分类器头来进行分类。fineturn 的 Loss 包含两项：第一项是 cluster 级，使原始图像和增强图像之间相同 cluster 的相似度尽量大，不同 cluster 之间的相似度尽量小。第二项是 instance 级别的，目的为拉近原始图像的 embedding 与增强图像的 embedding 之间的距离、扩大不同原始图像 embedding 之间的距离。 |
| [Nearest Neighbor Matching for Deep Clustering](https://openaccess.thecvf.com/content/CVPR2021/papers/Dang_Nearest_Neighbor_Matching_for_Deep_Clustering_CVPR_2021_paper.pdf) | NNM | CVPR 2021 | [Pytorch](https://github.com/ZhiyuanDang/NNM) | CIFAR10: 0.843  | NNM是一个分段式的方法，偏向缝合怪。它首先通过contrastive loss预训练backbone，如MoCo和SimCLR。在pretrain模型的基础上为每个样本p寻找若干个邻居Npre(N)（固定不变的）。而在训练过程中则是寻找local和global的邻居Nlocal(p)和Nglobal(p)（可变的）。NNM将这些邻居与原始样本作为正例集，通过不同的组合方式来构造正例对。instance层面最大化assignment的相似度，而class层面使用InfoNCE作为loss形式。聚类assignment通过多个head获得。 |
| [Prototypical Contrastive Learning of Unsupervised Representations](http://export.arxiv.org/pdf/2005.04966) | PCL | ICLR 2021 | [Pytorch](https://github.com/salesforce/PCL) | - | PCL是一个表征学习方法，它通过EM算法来迭代训练网络参数。在E步骤，通过K-Means算法计算prototypes，具体为取样本表征的均值，即聚类中心。在M步骤，通过最小化ProtoNCE loss来优化网络参数。两个步骤迭代执行。PCL采用了两分支的架构，其第二个分支与MoCo相同，也是使用了momentum更新策略。<br>EM算法的目的在于最小化一个对数似然函数。PCL证明了最小化ProtoNCE loss可以起到相同的作用，从而将PCL代入了EM算法的框架中。ProtoNCE loss可以分为两项：第一项是常规的InfoNCE loss，在训练的最初，它将起到warm-up的作用。第二项是对InfoNCE的修改版本，在分子中，正样本对由样本和它所属类的prototype构成；而分母中，则是选取了r个不同类的prototype作为负样本。值得注意的是，PCL重新设计了第二项中的温度参数，每一个类都有其对应的温度。若一个类中的样本距中心的平均距离越近，类中的样本点越多，则它对应的温度参数越小。PCL中提到这个设计还可以用来避免平凡解。 |
| [Refining Pseudo Labels with Clustering Consensus over Generations for Unsupervised Object Re-identification](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Refining_Pseudo_Labels_With_Clustering_Consensus_Over_Generations_for_Unsupervised_CVPR_2021_paper.pdf) | RLCC | CVPR 2021 | - | - | RLCC不是一个专门做聚类的方法，它只是借用了聚类的框架来进行Object Re-identification，其中的一些思想可以用作参考。首先，RLCC是基于generation的，每个generation的class num不一定相同——这是与聚类方法区别较大的一点。两个generation之间有一个class consensus matrix，即confusion matrix，只是这里行和列的维度不一定相同。RLCC通过两个generation的伪标签来统计构造matrix，矩阵的每一项的分子为两个类交集的样本数量，分母为并集的样本数量，最后再对每一行进行normalization。由此，新一轮的pseudo label由当前模型结果以及 前一轮的label（hard或soft）与matrix的相乘结果进行momentum的计算得到。 |
| [Jigsaw Clustering for Unsupervised Visual Representation Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Jigsaw_Clustering_for_Unsupervised_Visual_Representation_Learning_CVPR_2021_paper.pdf) | JigsawClustering | CVPR 2021 | [Pytorch](https://github.com/dvlab-research/JigsawClustering) | - | JigsawClustering是一个表征学习方法，与其他两分支数据增强对比学习方法不同，JigsawClustering另辟蹊径，从分割图像出发构造正样本对。具体而言，对于一个大小为n的batch，它将其中每个image分为4块，共4*n块，记录位置信息，在当前batch内随机充重组成n个image，通过backbone后再拆分为4*n个feature。重组再拆分看起来有一些多余，作者进行了实验来说明这对于效果有一定的提升。获得feature后，进行两种操作：1）通过MLP获得embedding，进行对比学习，分子部分为源自同一个image的块，分母部分为其他块。本质上它还是降每个image视为一类，没有考虑语义信息。2）通过一个FC，预测当前块属于原始图像中的哪个位置，这部分loss直接用CrossEntropy定义。两部分loss相加进行优化。 |