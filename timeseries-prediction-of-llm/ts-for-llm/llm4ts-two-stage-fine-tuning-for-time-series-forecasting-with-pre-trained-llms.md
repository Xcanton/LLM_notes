---
title: "llm4ts-two-stage-fine-tuning-for-time-series-forecasting-with-pre-trained-llms"
date: 2024-01-03T22:59:53+08:00
draft: false
---

---
description: https://arxiv.org/abs/2308.08469
---

# LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs

* 论文地址：[https://arxiv.org/abs/2308.08469](https://arxiv.org/abs/2308.08469)



## 概要

* 输入处理类似上一篇文章， 主要尝试解决了两个主要问题：
*
  * 如何将时间序列数据输入LLM？
  *
    * 将时间序列转为patch，并通过1d-conv层将每个时序patch转为gpt2 的输入维度大小；
    * 基于look-up 的patch location embedding;
    * 通过channel 独立(通过权重共享间接实现cross-channel交互）的patching， 训练时采用instance-norm，预测时采用RevIN;
    * 将每个patch内的第一个timestamp作为该patch的timestamp， patch内每个timestep的属性进行叠加作为该patch的属性；
  * 如何与现有的LLM进行集成？
  *
    * 时间序列进行自监督预训练(将LLM适配patch格式的时间序列数据）+下游预测任务微调；
    * 自监督训练时，冻结llm的self-attention和FFN，重新训练输入端和layerNorm;
    * 下游预测任务微调时，先进行linear probing(微调最后一层固定其他）,再微调所有参数的操作 ;
* 在多个公开数据集上的多变量时间序列预测，few-shot learing 和长时间序列预测超越专家网络模型。
