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
