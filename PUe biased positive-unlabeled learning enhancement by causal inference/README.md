### 1. Review of  PU Classification


- $ x \in \mathbb{R}^d $: Input sample in $ d $-dimensional real space.
- $ y \in \{+1, -1\} $: Corresponding labels.
- $ p_P(x) = p(x | y = +1) $: Probability distribution of positive data.
- $ p_N(x) = p(x | y = -1) $: Probability distribution of negative data.
- $ \mathcal{X}_P = \{x_i^P\}_{i=1}^{n_P} $: Set of positive data samples.
- $ \mathcal{X}_N = \{x_i^N\}_{i=1}^{n_N} $: Set of negative data samples.
- $ \pi = p(y = 1) $: Class prior probability.
- $ g : \mathbb{R}^d \to \mathbb{R} $: Binary classifier function.
- $ \theta $: Parameter of the classifier.
- $ \mathcal{L} : \mathbb{R} \times \{+1, -1\} \to \mathbb{R}_+ $: Loss function.
- $ R_P(g, +1) $: Expected risk loss of a positive sample.
- $ \mathbb{E}_{x \sim p_P(x)}[L(g(x, +1))] $: Expectation with respect to the positive data distribution $ p_P(x) $ of the loss function $ L $ evaluated at the classifier $ g $ with label $ +1 $.
- $ R_N(g, -1) $: Expected risk loss of a negative sample.
- $ \mathbb{E}_{x \sim p_N(x)}[L(g(x, -1))] $: Expectation with respect to the negative data distribution $ p_N(x) $ of the loss function $ L $ evaluated at the classifier $ g $ with label $ -1 $.



**传统PN二分类经验风险函数为**
$$
\hat{R}_{PN}(g) = \pi \hat{R}_P(g, +1) + (1 - \pi) \hat{R}_N(g, -1)
$$

其中：
$$
\hat{R}_P(g, +1) = \frac{1}{n_P} \sum_{i=1}^{n_P} L(g(x_i^P), +1)
$$
$$
\hat{R}_N(g, -1) = \frac{1}{n_N} \sum_{i=1}^{n_N} L(g(x_i^N), -1)
$$



**PU场景下的期望风险：**
$$
(1-\pi)R_N(g,-1) = R_U(g,-1)-\pi R_P(g,-1)
$$


未标记样本的期望风险$ R_U(g, -1) $ 定义为：
$$
R_U(g, -1) = \mathbb{E}_{x \sim p(x)}[L(g(x), -1)]
$$
因为$ p(x) = \pi p_P(x) + (1 - \pi) p_N(x) $，所以：
$$
R_U(g, -1) = \mathbb{E}_{x \sim \pi p_P(x) + (1 - \pi) p_N(x)}[L(g(x), -1)]
$$
根据期望的线性性质，这可以拆分为：
$$
R_U(g, -1) = \pi \mathbb{E}_{x \sim p_P(x)}[L(g(x), -1)] + (1 - \pi) \mathbb{E}_{x \sim p_N(x)}[L(g(x), -1)]
$$
换句话说：
$$
R_U(g, -1) = \pi R_P(g, -1) + (1 - \pi) R_N(g, -1)
$$
整理以得到$ R_N(g, -1) $：
$$
R_U(g, -1) - \pi R_P(g, -1) = (1 - \pi) R_N(g, -1)
$$
$$
R_N(g, -1) = \frac{R_U(g, -1) - \pi R_P(g, -1)}{1 - \pi}
$$

将上式代入传统二分类的期望风险，得到PU场景下的经验风险：
$$
\hat{R}_uPU(g) = \pi \hat{R}_P(g,+1)+\hat{R}_U(g,+1)-\pi \hat{R}_P(g,-1)
$$
