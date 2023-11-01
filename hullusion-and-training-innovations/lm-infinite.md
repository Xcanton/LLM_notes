---
title: "lm-infinite"
date: 2023-11-01T15:43:13+08:00
draft: false
---

# LM-Infinite

论文《[LM-INFINITE: SIMPLE ON-THE-FLY LENGTH GENERALIZATION FOR LARGE LANGUAGE MODELS](https://arxiv.org/abs/2308.16137)的阅读记录。

## Objective

针对开源大模型在长文本任务表现不佳的问题，提出对transformer的掩码与距离进行限制，提升模型在长文本生成内容的流畅度和相关性。

length generalization failure：生成长度超过训练文本长度（通常是预训练的平均长度）的回答。（the typical length in pre-training）
