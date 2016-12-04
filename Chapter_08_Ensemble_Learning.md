# 【《机器学习》第8章】集成学习

## 8.1 个体与集成

集成学习通过构建多个分类器并集成多个分类器来完成学习任务。具体步骤为：
1. 通过学习产生多个分类器；
2. 通过某些策略将这些分类器进行结合。
需要注意的是，如果分类器种类相同，则集成是同质的`homogeneous`，学习器称为基学习器，相应的学习算法称为基学习算法；如果分类器种类不相同，则集成是异质的`heterogenous`，学习器称为组件学习器。

集成学习的目的是通过组合多个分类器来获得更好的**泛化性能**。但是从常识上来看，好的东西和坏的东西掺杂起来通常会获得不好不坏的东西。集成学习为什么能通过集成来获得泛化能力更强的分类器呢？原因在于，在理想情况下，学习器器选取的准则为**多样性**和**准确性**（具体可见图8.2的举例）。用公式描述为：对二分类问题，$y\in \{-1,+1\}$，真实函数$f$，假定分类器的错误率为$\epsilon$，即对每个分类器有$$P\big(h(\mathbf{x})\neq f(\mathbf{x})\big)=\epsilon$$，若有超过半数的分类器分类正确，则集成分类就正确$$H(\mathbf{x})=sign\Big(\sum_{i=1}^Th_i(\mathbf{x})\Big)$$，**假设分类器的错误率相互独立**，则由`Hoeffding不等式`可知，集成的错误率为$$P\Big(H(\mathbf{x})\neq f(\mathbf{x})\Big)=\sum_{k=0}^{\lfloor T/2 \rfloor}C_T^k(1-\epsilon)^k\epsilon^{T-k}\\ \le \exp\Big(-\frac{1}{2}T(1-2\epsilon)^2\Big)$$。当$T$不断增大时，错误率指数级减小。

但是需要指出的一个重要前提条件是，**假设分类器的错误率相互独立**。这个条件在实际中很难成立，所以学习器的“准确性”和“多样性”本身就存在冲突。如何产生并结合好而不同的学习器是集成学习的一个核心研究内容。

一般情况下，集成学习分为可以序列化生成学习器的Boosting方法和并行生成学习器的Bagging方法、随机森林Random Forest方法。

## 8.2 Boosting

本节的Boosting方法以AdaBoosting为例，虽然之前阅读过AdaBoosting的相关文献，但是本节的内容还是花费了一个晚上的时间去梳理。
从整体上来看，AdaBoosting每一轮迭代是以上一轮迭代的结果为基础，重点针对上一轮迭代得到的学习器的错误分类样本进行训练。这就和之前提到的学习器的选取准则准确性和多样性形成了对应关系。
图8.3是AdaBoosting的算法，容易产生疑问的是第6行的学习器权重更新公式和第7行的样本分布更新公式。本节重点说明并进行证明的也正是这两处。
在证明上述两个公式之前，需要对AdaBoosting所使用的指数损失函数做一个说明，因为指数损失函数是之后证明过程的基础。
$$\mathcal{l}_{exp}=E_{\mathbf{x}\sim\mathcal{D}}\Big(e^{-f(\mathbf{x})H(\mathbf{x})}\Big)$$
对$H(\mathbf{x})$偏导，得$$\frac{\partial\mathcal{l}_{exp}}{\partial H(\mathbf{x})}=e^{-f(\mathbf{x})H(\mathbf{x})}(-f(\mathbf{x}))\\=-e^{-H(\mathbf{x})}P(f(\mathbf{x}=1|\mathbf{x}))+e^{H(\mathbf{x})}P(f(\mathbf{x})=-1|\mathbf{x})$$，令上式为0，得$$H(\mathbf{x})=\frac{1}{2}\ln\frac{P(f(\mathbf{x})=1|\mathbf{x})}{P(f(\mathbf{x})=-1|\mathbf{x})}$$。即有$$sign(H(\mathbf{x}))=\arg\min_{y\in\{-1,1\}}P(f(\mathbf{x})=y|\mathbf{x})$$，$sign(H(\mathbf{x}))$达到了`贝叶斯最优错误率`，可以用来替代0/1损失函数，并且具有连续、可导的优点。

### 学习器权重更新公式证明

假设在第t轮训练中得到的学习器为$h_t(\mathbf{x})$，则当前的分类器权重$\alpha_t$应使
$$\mathcal{l}_{exp}(\alpha_th_t(\mathbf{x})|\mathcal{D_t})=E_{\mathbf{x}\sim\mathcal{D_t}}[e^{-f(\mathbf{x})h_t(\mathbf{x})\alpha_t}]$$最小。将上式对$\alpha_t$求偏导得$$\frac{\partial \mathcal{l}_{exp}(\alpha_th_t(\mathbf{x})|\mathcal{D_t})}{\partial \alpha_t}=-e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t$$。令上式为$0$，得$$\alpha_t=\ln\frac{1-\epsilon_t}{\epsilon_t}$$，即为第6行中的学习器权重公式。

### 样本分布更新公式证明

在获取$H_{t-1}$之后，AdaBoosting对样本分布进行一些调整，使得在第$t$轮训练中得到的学习器能够针对$t-1$轮训练中的错分样本进行正确分类。即最小化$$\mathcal{l}_{exp}(H_{t-1}+h_t|\mathcal{D})=E_{\mathbf{x}\sim\mathcal{D}}\Big[e^{-f(\mathbf{x})H_{t-1}(\mathbf{x})}e^{-f(\mathbf{x})h_t(\mathbf{x})}\Big]$$。
由于有$f(\mathbf{x})^2=h_t(\mathbf{x})^2=1$，则上式泰勒展开式近似为$$\mathcal{l}_{exp}(H_{t-1}+h_t|\mathcal{D})\simeq E_{\mathbf{x}\sim\mathcal{D}}\Big[e^{-f(\mathbf{x})H_{t-1}(\mathbf{x})}\Big(1-f(\mathbf{x})h_{t}(\mathbf{x})+\frac{1}{2}\Big)\Big]$$。
理想的学习器$$h_t(\mathbf{x})=\arg\max_hE_{\mathbf{x}\sim\mathcal{D}}\Big[e^{-f(\mathbf{x})H_{t-1}(\mathbf{x})}f(\mathbf{x})h(\mathbf{x})\Big]\\=\arg\max_hE_{\mathbf{x}\sim\mathcal{D}}\Big[\frac{e^{-f(\mathbf{x})H_{t-1}(\mathbf{x})}}{E_{\mathbf{x}\sim \mathcal{D}}[e^{-f(\mathbf{x})H_{t-1}(\mathbf{x})}]}f(\mathbf{x})h(\mathbf{x})\Big]$$将在分布$\mathcal{D}_t$下最小化分类误差。
令$\mathcal{D}_t$表示一个分布，$$\mathcal{D}_{t+1}(\mathbf{x})=\frac{\mathcal{D}(\mathbf{x})e^{-f(\mathbf{x})H_t(\mathbf{x})}}{E_{\mathbf{x}\sim\mathcal{D}}[e^{-f(\mathbf{x})H_t(\mathbf{x})}]}=\mathcal{D}_te^{e^{-f(\mathbf{x})\alpha_t h_t(\mathbf{x})}}\frac{E_{\mathbf{x}\sim\mathcal{D}}[e^{-f(\mathbf{x})H_{t-1}(\mathbf{x})}]}{E_{\mathbf{x}\sim\mathcal{D}}[e^{-f(\mathbf{x})H_t(\mathbf{x})}]}$$，第7行样本分布更新公式得证。

## 8.3 Bagging与随机森林

### 8.3.1 Bagging

Bagging是并行集成学习方法最著名的代表。Bagging方法首先采样出$T$个含有$m$个样本的样本子集，然后在每个样本子集上分别训练出一个个体学习器，然后将这些学习器进行结合。如果采用`自助采样法`获得$m$个样本子集，则大约有63.2%的样本包含在样本子集中，剩余的36.8%的样本可以用来对基学习器进行`包外估计(out-of-bag estimation)`。从`偏差-方差分解`的角度来看，Bagging重点关注降低方差，因此它不在易受样本扰动的决策树、神经网络等基学习器上效果更加明显。

### 8.3.2 随机森林

随机森林在Bagging基础上引入了随机属性选择。但是与Bagging中基学习器“多样性”仅仅依靠样本扰动不同，随机森林的“多样性”还来自于属性扰动。这就使得最终集成的泛化性能通过个体学习器之间的差异性的增加而进一步提升。

但是同时需要注意的是，随机森林的起始性能较差，随着个体学习器的数量增加，随机森林会收敛到更低的泛化误差。

## 8.4 结合策略

### 8.4.1 平均法

平均法分为简单平均法和加权平均法。其中，加权平均法的权重是从训练数据中学习而得，现实中由于数据存在误差和噪声，这将使得学出的权重不完全可靠。一般而言，在个体学习器性能差异较大时使用加权平均法，而在个体学习器性能相近时使用简单平均法。

### 8.4.2 投票法

投票发分为绝对多数投票法、相对多数投票法和加权投票法。

在一些能够同时输出分类标签和分类置信度的学习器，将置信度转化为类概率值虽然不太准确，但是能够取得比直接使用分类标签更好的结合性能。

### 8.4.3 学习法

学习法是指使用学习方法得到各个学习器的权重并进行结合。

Stacking方法为了避免过拟合，通常使用交叉验证法或者留一法，用训练初级学习器未使用到的数据来训练次级学习器。次级学习器的输入属性和次级学习器算法对Stacking集成的泛化性能有很大影响。有研究表明，将初级学习器的输出类概率作为次级学习器的输入属性，用`多响应线性回归(Multi-response Liner Regression, MLR)`作为次级学习算法效果较好，在MLR中使用不同的属性集更佳。

`贝叶斯模型平均`基于后验概率来为不同模型赋予权重，可视为加权平均法的一种特殊实现。

## 8.5 多样性

### 8.5.1 误差-分歧分解

8.2节中提到过，个体学习器应该“好而不同”，本节主要针对这一点做一个简要的分析.

对于示例$\mathbf{x}$，定义学习器$h_i$的“分歧”(ambuiguity)为
$$A(h_i|\mathbf{x})=(h_i(\mathbf{x})-H(\mathbf{x}))^2$$
则集成的分歧为
$$A(H(\mathbf{x})|\mathbf{x})=\sum_{i=1}^Tw_i(h_i(\mathbf{x})-H(\mathbf{x}))^2$$
上述两项分歧主要定义了个体学习器在样本$\mathbf{x}$上的不一致性.

个体学习器$h_i$和集成学习器$H$在$\mathbf{x}$上的误差定义分别为
$$E(h_i(\mathbf{x})|\mathbf{x})=(h_i(\mathbf{x})-f(\mathbf{x}))^2$$
$$E(H(\mathbf{x})|\mathbf{x})=(H(\mathbf{x})-f(\mathbf{x}))^2$$

则有
$$\bar{A}(h|\mathbf{x})=\sum_{i=1}^Tw_iE(h_i|\mathbf{x})-E(H|\mathbf{x})$$

将上式推广到全样本控件,设样本概率分布为$p(\mathbf{x})$,则有
$$\sum_{i=1}^Tw_i\lmoustache A(h_i|\mathbf{x})p(\mathbf{x})d\mathbf{x}=\sum_{i=1}^Tw_i\lmoustache E(h_i|\mathbf{x})p(\mathbf{x})d\mathbf{x}-\lmoustache E(H|x)p(\mathbf{x})d\mathbf{x}$$
令个体误差和分歧为
$$E_i=\lmoustache E(h_i|\mathbf{x})p(\mathbf{x})d\mathbf{x}$$
$$A_i=\lmoustache A(h_i|\mathbf{x})p(\mathbf{x})d\mathbf{x}$$
令集成误差为
$$\bar{E}=\sum_{i=1}^T\lmoustache E(H|\mathbf{x})p(\mathbf{x})d\mathbf{x}$$
再令$\bar{E}$和$\bar{A}$分别为个体学习器的加权误差值和加权分歧值,则有
$$E=\bar{E}-\bar{A}$$
从上式可以看出,个体学习器准确性越高($\bar{E}$越大),多样性越大($\bar{A}$越大),则集成越好.

但是在现实中很难针对$\bar{E}-\bar{A}$进行优化,不仅因为其是在全样本空间上的定义,还由于$\bar{A}$是一个不可直接操作的准确性度量.

另外需要注意的是,本节中的推导过程是针对回归问题的,不能推广到分类问题上去.

### 8.5.2 多样性度量

多样性度量是用于度量集成中个体分类器的多样性,通常考虑分类器间两两不相似性.常见度量准则有(公式略)
- 不合度量(disagreement measure)
- 相关系数(correlation coefficient)
- Q-统计量(Q-statistic)
- $\kappa$-统计量($\kappa$-statistic)

由于上述统计量都是成对学习器的度量值,可以通过二位坐标平面进行展示.其中$x$轴是这对分类器的多样性度量值,纵坐标轴是他们的平均误差.通常情况下,点云位置越高,个体学习器的准确性越低,点云位置越靠右,个体学习器多样性越小.

### 8.5.3 多样性增强

为了增强多样性,一般思路是在学习过程中增加扰动,即进入随机性.常见做法有对数据样本,输入属性,输出表示,算法参数进行扰动.

#### 数据样本扰动

给定初始训练样本集,从中产生不同的数据子集,在子集上对学习器进行训练.需要注意的是,有些类型的学习器比较容易受到训练样本的扰动,如决策树,神经网络等;有些类型的学习器不容易受到训练样本的扰动,如线性学习器,支持向量机,朴素贝叶斯,k近邻学习器等,这样的学习器称为`稳定基学习器(stable base learner)`.对于稳定基学习器往往需要使用其他类型的扰动.

#### 输入属性扰动

训练样本通常使用一组属性进行描述,属性扰动是从这一组属性中随机采样出属性子集对学习器进行训练.如果属性类别比较多,则可以再一定程度上降低运算复杂度;如果属性类别较少,则不宜使用输入属性扰动.

#### 输出表示扰动

对输出表示进行操纵以增强多样性.举例有`翻转法`,`输出调制法`,`ECOC法`.

#### 算法参数扰动

基学习算法一般都有参数需要进行设置,通过随机设置不同的参数(如神经网络隐神经元数目),往往可以产生差异较大的个体学习器.
