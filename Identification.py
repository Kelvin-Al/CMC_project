import numpy as np
from scipy import stats
from typing import List, Dict, Any, Callable
from Tools import *
from Models.CMC_base import *

np.random.seed(42)
def calculate_posterior_heterogeneity(data_groups, prior_mean=0.0, prior_variance=1.0, likelihood_variance=None):
    """
    计算后验异质性，同时返回原始方法和Scott标准化方法的结果

    参数:
    data_groups: 数据组列表，每个元素是一个numpy数组
    prior_mean: 先验均值
    prior_variance: 先验方差
    likelihood_variance: 似然方差（如果为None，则从数据估计）

    返回:
    包含原始异质性度量和Scott标准化异质性度量的字典
    """
    k = len(data_groups)
    if k < 2:
        raise ValueError("至少需要2个数据组来计算异质性")

    # 1. 估计似然方差（如果未提供）
    if likelihood_variance is None:
        total_ss = 0
        total_df = 0
        for i in range(k):
            data = data_groups[i]
            if len(data) > 1:
                total_ss += np.sum((data - np.mean(data)) ** 2)
                total_df += len(data) - 1
        likelihood_variance = total_ss / total_df if total_df > 0 else 1.0

    # 2. 计算每组的后验统计量
    posterior_means = []
    posterior_variances = []
    sample_sizes = []
    data_means = []

    prior_precision = 1.0 / prior_variance

    for i in range(k):
        data = data_groups[i]
        n_i = len(data)

        if n_i == 0:
            raise ValueError(f"第{i}组数据为空")

        # 样本统计量
        x_bar_i = np.mean(data)
        data_means.append(x_bar_i)
        sample_sizes.append(n_i)

        # 计算后验参数
        likelihood_precision = n_i / likelihood_variance
        posterior_precision = prior_precision + likelihood_precision
        posterior_variance_i = 1.0 / posterior_precision

        # 后验均值
        posterior_mean_i = (prior_precision * prior_mean +
                            likelihood_precision * x_bar_i) / posterior_precision

        posterior_means.append(posterior_mean_i)
        posterior_variances.append(posterior_variance_i)

    # 转换为numpy数组
    mu_hat = np.array(posterior_means)
    sigma2_hat = np.array(posterior_variances)
    weights = 1.0 / sigma2_hat

    # 3. 计算加权总体均值（Scott共识估计）
    mu_pooled = np.sum(weights * mu_hat) / np.sum(weights)

    # ========== 原始方法（旧定义） ==========
    Q_original = np.sum(weights * (mu_hat - mu_pooled) ** 2)

    # 调整因子
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights ** 2)
    c = sum_w - sum_w2 / sum_w

    # 组间方差τ²
    if Q_original > (k - 1):
        tau2 = max(0, (Q_original - (k - 1)) / c)
    else:
        tau2 = 0.0

    # I²指数
    if Q_original > 0:
        I2_original = max(0, (Q_original - (k - 1)) / Q_original) * 100
    else:
        I2_original = 0.0

    # p值
    if k > 1:
        p_value_original = 1 - stats.chi2.cdf(Q_original, k - 1)
    else:
        p_value_original = np.nan

    # ========== Scott标准化方法（新定义） ==========
    # 计算标准化残差
    z_scores = (mu_hat - mu_pooled) / np.sqrt(sigma2_hat)

    # 计算Scott异质性统计量
    Q_scott = np.sum(z_scores ** 2)  # 服从 χ²(k)

    # 期望值（无异质性假设下）
    expected_Q = k

    # 计算超额异质性比例
    if expected_Q > 0:
        excess_heterogeneity = max(0, (Q_scott - expected_Q) / expected_Q) * 100
    else:
        excess_heterogeneity = 0

    # 计算p值
    p_value_scott = 1 - stats.chi2.cdf(Q_scott, k)

    # 计算标准化残差的描述统计
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores, ddof=1)
    z_max_abs = np.max(np.abs(z_scores))

    # ========== 返回结果 ==========
    return {
        # 基本信息
        'num_groups': k,
        'prior_mean': prior_mean,
        'prior_variance': prior_variance,
        'likelihood_variance': likelihood_variance,

        # 原始数据
        'data_means': data_means,
        'sample_sizes': sample_sizes,

        # 后验统计量
        'posterior_means': posterior_means,
        'posterior_variances': posterior_variances,
        'weights': weights.tolist(),
        'pooled_mean': mu_pooled,

        # ===== 原始方法结果 =====
        'original': {
            'Q': Q_original,
            'I2': I2_original,
            'tau2': tau2,
            'p_value': p_value_original,
            'method': '原始方法（评估原始后验均值差异）'
        },

        # ===== Scott标准化方法结果 =====
        'scott': {
            'Q': Q_scott,
            'expected_Q': expected_Q,
            'excess_heterogeneity': excess_heterogeneity,
            'p_value': p_value_scott,
            'z_scores': z_scores.tolist(),
            'z_mean': z_mean,
            'z_std': z_std,
            'z_max_abs': z_max_abs,
            'scott_I2': max(0, (Q_scott - k) / Q_scott) * 100 if Q_scott > 0 else 0,
            'method': 'Scott标准化方法（评估加权后的标准化残差）'
        },

        # 解释性指标
        'interpretation': {
            'scott_reliable': p_value_scott > 0.05,  # Scott方法是否可靠
            'significant_heterogeneity_original': p_value_original < 0.05,
            'significant_heterogeneity_scott': p_value_scott < 0.05,
            'recommendation': '使用Scott方法' if p_value_scott > 0.05 else '警告：存在显著异质性'
        }
    }





def compare_heterogeneity_with_prior(data_groups: np.ndarray):
    """比较不同先验下的异质性"""

    # 测试用例：您的两个例子
    print("=" * 60)
    print("测试不同先验下的异质性")
    print("=" * 60)

    # 先验：N(0, 1)
    prior_mean = 0.0
    prior_variance = 1.0
    likelihood_variance = 10.0  # 假设已知

    # 情况A：组1在100附近，组2在99附近
    print("\n情况A：组1~N(100,1), 组2~N(99,1)")
    print("-" * 40)

    # 生成数据
    np.random.seed(42)
    mean = np.random.normal(loc = 100, scale = 0.3, size = 1000)
    std = [10]+[1]*99
    std = np.array(std,dtype=float)
    size = np.array([100]*100)
    data_A = generate_heterogeneous_data(100, mean, std, size)
    results_A = calculate_posterior_heterogeneity(
        data_A, prior_mean, prior_variance, likelihood_variance
    )


    print(f"数据均值: {results_A['data_means']}")
    print(f"后验均值: {[f'{x:.4f}' for x in results_A['posterior_means']]}")
    print(f"后验异质性 I² = {results_A['scott']['scott_I2']:.2f}% Ex_heter = {results_A['scott']['excess_heterogeneity']:.2f}%")
    print(f"Q = {results_A['scott']['Q']:.4f}, p = {results_A['scott']['p_value']:.4f}")

    # 情况B：组1在-0.5附近，组2在0.5附近
    print("\n情况B：组1~N(-0.5,1), 组2~N(0.5,1)")
    print("-" * 40)

    data_B = [
        np.random.normal(-0.05, 1, 100000),
        np.random.normal(0.05, 1, 100000)
    ]

    results_B = calculate_posterior_heterogeneity(
        data_B, prior_mean, prior_variance, likelihood_variance
    )

    print(f"数据均值: {results_B['data_means']}")
    print(f"后验均值: {[f'{x:.4f}' for x in results_B['posterior_means']]}")
    print(f"后验异质性 I² = {results_B['scott']['scott_I2']:.2f}% Ex_heter = {results_B['scott']['excess_heterogeneity']:.2f}%")
    print(f"Q = {results_B['scott']['Q']:.4f}, p = {results_B['scott']['p_value']:.4f}")


import numpy.typing as npt
from scipy.stats import entropy

def hierarchical_consensus_diagnostic(
        group_data: List[npt.NDArray],
        total_sample_size: int,
        metric: str
) -> Dict[str, Any]:
    """
    通过分层抽样比较整体共识与分层共识的差异来诊断异质性

    参数:
        group_data: 分组数据列表，每个元素是一个ndarray代表一组数据
        total_sample_size: 分层抽样的总数据量
        run_mcmc_on_data: 函数，接收数据返回后验分布样本
        consensus_aggregator: 函数，接收多个后验分布样本返回共识后验
        metrics: 使用的比较指标列表

    返回:
        诊断结果字典
    """

    def stratified_sampling(data_list: List[npt.NDArray], total_n: int) -> List[npt.NDArray]:
        """按组大小比例进行分层抽样"""
        # 计算每组应有的样本量（按比例分配）
        group_sizes = [len(data) for data in data_list]
        total_original_size = sum(group_sizes)
        sample_sizes = [int(total_n * size / total_original_size) for size in group_sizes]

        # 确保总样本量正确（处理整数舍入误差）
        diff = total_n - sum(sample_sizes)
        if diff > 0:
            # 将剩余样本量分配给数据量较大的组
            largest_groups = np.argsort(group_sizes)[-diff:]
            for i in largest_groups:
                sample_sizes[i] += 1

        # 执行分层抽样
        sampled_data = []
        for i, data in enumerate(data_list):
            if sample_sizes[i] > 0:
                indices = np.random.choice(len(data), sample_sizes[i], replace=False)
                sampled_data.append(data[indices])
            else:
                sampled_data.append(np.array([]))  # 空组

        return sampled_data

    def compute_metric(dist1: npt.NDArray, dist2: npt.NDArray, metric = metric) -> float:
        """计算两个分布之间的差异度量"""
        if metric == 'kl_divergence':
            # 使用直方图近似计算KL散度
            hist1, bins = np.histogram(dist1, bins=50, density=True)
            hist2, _ = np.histogram(dist2, bins=bins, density=True)
            # 避免除零错误
            hist1 = np.clip(hist1, 1e-10, None)
            hist2 = np.clip(hist2, 1e-10, None)
            return entropy(hist1, hist2)

        elif metric == 'wasserstein':
            # 计算Wasserstein距离（推土机距离）
            from scipy.stats import wasserstein_distance
            return wasserstein_distance(dist1, dist2)

        elif metric == 'mean_diff':
            # 均值差异（标准化）
            std_pooled = np.sqrt((np.var(dist1) + np.var(dist2)) / 2)
            return np.abs(np.mean(dist1) - np.mean(dist2)) / (std_pooled + 1e-10)

        elif metric == 'js_divergence':
            # Jensen-Shannon散度（对称的KL变体）
            hist1, bins = np.histogram(dist1, bins=50, density=True)
            hist2, _ = np.histogram(dist2, bins=bins, density=True)
            hist1 = np.clip(hist1, 1e-10, None)
            hist2 = np.clip(hist2, 1e-10, None)
            m = 0.5 * (hist1 + hist2)
            return 0.5 * (entropy(hist1, m) + entropy(hist2, m))

        else:
            raise ValueError(f"不支持的度量标准: {metric}")

    # 主逻辑开始
    k = len(group_data)

    # 1. 分层抽样
    sampled_groups = stratified_sampling(group_data, total_sample_size)

    # 2. 策略A: 整体共识（合并所有抽样数据）
    combined_data = np.concatenate([data for data in sampled_groups if len(data) > 0])
    base_mcmc = BaseMCMC(n_samples=2000, n_burnin=1000)
    posterior_overall = base_mcmc.run(combined_data)

    consensus_mcmc = ConsensusMCMC(n_workers=k, n_samples=2000, n_burnin=1000)
    posterior_hierarchical = consensus_mcmc.run_consensus(group_data)

    results = {}
    results[metric] = compute_metric(posterior_overall, posterior_hierarchical)
    return results


# 运行测试
if __name__ == "__main__":
    #compare_heterogeneity_with_prior(None)
    group_data = generate_heterogeneous_data(5, [100,100,100,100,110], [1,1,1,1,1], [2000,3000,4000,1000,500])
    results = hierarchical_consensus_diagnostic(group_data,1050,'kl_divergence')
    print(results)
