---
description: https://arxiv.org/abs/2302.11939
---

# One Fits All: Power General Time Series Analysis by Pretrained LM

* 论文地址：[https://arxiv.org/abs/2302.11939](https://arxiv.org/abs/2302.11939)
* 论文代码地址：[https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)

## 概要

* 输入端先进行先对序列进行RevIN(instanceNorm), 缓解分布漂移；
* 再参考PatchTST， 将时间序列切割为片段处理，每个片段有postion embedding;
* 去除token embedding层，将切分的片段输入linear层转为模型需要的输入维度；
* 冻结编码知识的 multi-head 和FFN层，微调LayerNorm和位置编码；
* 文章认为基于文本域训练的llm适用于时间序列的一个可能原因是， 自监督的self-attention模块在训练过程中学会了和和具体数据无关的一些运算规则，比如PCA（通过对比对比两个模块的中间结果），使之成为一个广义的计算引擎；
* 以GPT2为backbone进行了时间序列异常检测、长短期预测等实验，效果均较好;

