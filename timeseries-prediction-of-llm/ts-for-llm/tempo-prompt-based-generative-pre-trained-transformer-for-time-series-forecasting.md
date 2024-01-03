---
title: "tempo-prompt-based-generative-pre-trained-transformer-for-time-series-forecasting"
date: 2024-01-03T22:59:53+08:00
draft: false
---

---
description: https://arxiv.org/abs/2310.04948
---

# TEMPO: prompt-based generative pre-trained transformer for time series forecasting

* 论文地址：[https://arxiv.org/abs/2310.04948](https://arxiv.org/abs/2310.04948)

## 任务定义-序列预测任务

$$
{\hat x}^i_{t},{\hat x}^i_{t+1},\ldots,{\hat x}^i_{t+H-1}=F({x}^i_{t-K},{x}^i_{t-K+1},\ldots,{x}^i_{t-1};{V_i};	\Phi)
$$

* K是特征窗口长度，H是预测序列长度，通过过去K个序列值，预测下H个值
* 当需要预测多个特征时，i可以为不同的特征编号
* $V\_i$是对于特征V的prompt，$\phi$是模型参数

与传统文本生成任务不同，Tempo按照时序预测的任务形式构成输入prompt，通过先验规则进行特征工程。

## 加入序列先验知识

### 时序数据构成

$$
X_i=X^i_T + X^i_S+X^i_R
$$

* 将原数据划分成长期特征、季节性特征和残差特征，一个输入数据由对应的三个特征值相加

#### 长期特征（Trend）

$$
X_T \in \mathbb{R}^{n \times L}=\frac {1} {2k+1} \sum^k_{j=-k}(X_{i+j})
$$

#### 季节性特征（Seasonal，采用局部加权移动平均）

$$
{Lowess\ Smoother}()
$$

#### 残差项

$$
X^i_R=X_i-X^i_T - X^i_S
$$

### 正则化方式

#### 数据构成后

$$
{\hat x}_{Tt}^i = \gamma_T(x_{Tt}^i-\mathbb{E}[x_{Tt}^{i}]/\sqrt{Var[x_{Tt}^i]+\epsilon_T})+\beta_T
$$

* $\mathbb{E}\[x\_{Tt}^{i}]$是平均数
* $Var\[x\_{Tt}^i]$是标准差
* $\gamma\_T$和$\beta\_T$是可学习的偏移参数

在构成数据之后，每个组成部分都进行reverse instance normalization，能够（促进知识转移，并最大程度得减小转移损失？）。但相对应的，在输出前需要逆向标准化回来。

#### 预测后

$$
{\widehat Y}_{*t}^i=\sqrt{Var[x_{Tt}^i]+\epsilon_*} \cdot {(\frac{Y_{*t}^i-\beta_*} {\gamma_*})+\mathbb{E}[x_{Tt}^{i}]}
$$

### Patching

对三个组成部分常规Patch

$$
\mathcal{P}_T^i \in 
\mathbb{R}^{L_P \times N}
$$

$$
N = \lfloor \frac {(L-L_P)} {S} \rfloor + 2
$$

### 特征池召回

除了时序信息以外，论文还从已有特征池（key是特征，value是预先写好的prompt文本）中根据不同组成部分召回相近的特征表达。

$$
V_K = {\{(k_1,V_1),(k_2,V_2), \ldots, (k_M, V_M)\}}
$$

* M是特征池的大小

对应的，针对给定的patch序列和特征的相似度计算公式为：

$$
\gamma(\mathcal{P}_T^i,k_m)=\mathcal{P}_T^i \cdot k_m / \Vert \mathcal{P}_T^i\Vert \Vert k_m\Vert
$$

模型将近似度选择的embedding一起end2end训练，对每个query的 $\mathcal{P}\_T^i$ 召回 Top-K 个相近的prompt，并且拼接在query序列的前面。所以最后对每个组成部分的输入为：

$$
x_T=[V_{s1}; \ldots;V_{s \mathcal{K}};\mathcal{P}_T], 1 \leq \mathcal{K} \leq M
$$

## GPT输入

将三个组成部分拼接起来输入，但是文章也给出了一个替代方案：对每个组成部分单独输入GPT，将输出的向量反向标准化后加和。

## 可解释性（Interpretability）

假设模型的三个组成部分跟输出的向量具有非线性的关系，可以通过GAM（generalized additive model）建模

$$
g(Y)=F_{\not o}+\sum_i{F_i(x_i)}+\sum_t{F_{\mathcal{I}_t}(x_{\mathcal{I}_t})}
$$

* $\mathcal{I}\_t$是多个组成部分交互的集合（a set of multiple instract components）
* $F\_{\not o}$是归一化常数
* i是各个组成部分

然后可以通过一阶系数（first-order sentitivity index）或者SHAP值（SHapley Additive exPlanations）来衡量不同组成部分的敏感性。
