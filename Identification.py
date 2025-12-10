import numpy as np
from scipy import stats
from typing import Dict


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


def assess_practical_heterogeneity_from_data(data_groups, prior_mean=0.0, prior_variance=1.0,
                                             likelihood_variance=None, effect_size_threshold=0.2,
                                             threshold_type='cohen'):
    """
    直接从数据出发，基于效应量阈值评估实际异质性

    参数:
    data_groups: 数据组列表，每个元素是一个numpy数组
    prior_mean: 先验均值
    prior_variance: 先验方差
    likelihood_variance: 似然方差（如果为None，则从数据估计）
    effect_size_threshold: 实际重要的最小效应量
    threshold_type: 阈值类型
        - 'cohen': Cohen's d (标准化的效应量)
        - 'raw': 原始尺度阈值
        - 'relative': 相对于合并标准差的百分比

    返回:
    包含实际异质性评估的字典
    """
    import numpy as np
    from scipy import stats

    k = len(data_groups)
    if k < 2:
        raise ValueError("至少需要2个数据组")

    # 1. 计算基本统计量
    # 估计似然方差（如果未提供）
    if likelihood_variance is None:
        total_ss = 0
        total_df = 0
        for i in range(k):
            data = data_groups[i]
            if len(data) > 1:
                total_ss += np.sum((data - np.mean(data)) ** 2)
                total_df += len(data) - 1
        likelihood_variance = total_ss / total_df if total_df > 0 else 1.0

    # 计算每组的后验统计量
    posterior_means = []
    posterior_variances = []
    data_means = []
    sample_sizes = []

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

    # 2. 计算Scott共识估计
    weights = 1.0 / sigma2_hat
    mu_pooled = np.sum(weights * mu_hat) / np.sum(weights)

    # 3. 计算原始Scott异质性（作为基准）
    z_scores = (mu_hat - mu_pooled) / np.sqrt(sigma2_hat)
    Q_scott = np.sum(z_scores ** 2)
    p_value_scott = 1 - stats.chi2.cdf(Q_scott, k)

    # 4. 计算效应量相关的统计量
    pooled_sd = np.sqrt(likelihood_variance)

    # 计算各组间的最大效应量（Cohen's d）
    max_mean_diff = np.max(mu_hat) - np.min(mu_hat)
    cohens_d_max = max_mean_diff / pooled_sd

    # 计算各组与共识的效应量
    effect_sizes_vs_pooled = np.abs(mu_hat - mu_pooled) / pooled_sd

    # 5. 根据阈值类型确定实际阈值
    if threshold_type == 'cohen':
        # Cohen's d阈值
        practical_threshold = effect_size_threshold  # 例如0.2
        threshold_in_raw_scale = practical_threshold * pooled_sd
    elif threshold_type == 'raw':
        # 原始尺度阈值
        practical_threshold = effect_size_threshold
        threshold_in_raw_scale = practical_threshold
    elif threshold_type == 'relative':
        # 相对于合并标准差的百分比
        practical_threshold = effect_size_threshold / 100.0  # 例如0.05表示5%
        threshold_in_raw_scale = practical_threshold * pooled_sd
    else:
        raise ValueError("threshold_type必须是'cohen'、'raw'或'relative'")

    # 6. 应用阈值：只考虑超过阈值的差异
    abs_diffs = np.abs(mu_hat - mu_pooled)

    # 标记哪些差异是"实际重要的"
    is_practically_important = abs_diffs > threshold_in_raw_scale

    # 计算"实际重要"的差异
    # 对于小于阈值的差异，我们将其视为0（不贡献异质性）
    effective_diffs = np.where(is_practically_important, mu_hat - mu_pooled, 0)

    # 7. 计算阈值调整后的统计量
    # 调整后的z-scores：小于阈值的差异视为0
    z_scores_adjusted = effective_diffs / np.sqrt(sigma2_hat)
    Q_scott_adjusted = np.sum(z_scores_adjusted ** 2)

    # 计算"实际重要"的异质性比例
    # Q_practical: 仅来自超过阈值的差异
    Q_practical = np.sum(z_scores_adjusted ** 2)
    practical_heterogeneity_ratio = Q_practical / Q_scott if Q_scott > 0 else 0

    # 计算阈值调整后的p-value
    p_value_adjusted = 1 - stats.chi2.cdf(Q_scott_adjusted, k)

    # 8. 计算"实际重要"的组数比例
    n_important = np.sum(is_practically_important)
    proportion_important = n_important / k

    # 9. 计算最大"实际重要"效应量
    if n_important > 0:
        important_effect_sizes = effect_sizes_vs_pooled[is_practically_important]
        max_practical_effect = np.max(important_effect_sizes) if len(important_effect_sizes) > 0 else 0
    else:
        max_practical_effect = 0

    # 10. 综合评估
    # 判断是否具有实际重要性
    has_practical_importance = (n_important > 0) and (max_practical_effect > effect_size_threshold)

    # 评估建议
    if not has_practical_importance:
        recommendation = "✅ 无异质性：所有差异都在实际不重要范围内"
        risk_level = "低"
    elif proportion_important < 0.3 and max_practical_effect < 0.5:
        recommendation = "⚠️  轻微异质性：少数组有较小实际差异"
        risk_level = "中低"
    elif proportion_important < 0.5 and max_practical_effect < 0.8:
        recommendation = "⚠️  中等异质性：部分组有明显实际差异"
        risk_level = "中"
    else:
        recommendation = "❌ 严重异质性：多数组有较大实际差异"
        risk_level = "高"

    return {
        # 基本信息
        'num_groups': k,
        'pooled_sd': pooled_sd,
        'pooled_mean': mu_pooled,

        # 原始统计量（作为参考）
        'original': {
            'Q_scott': Q_scott,
            'p_value': p_value_scott,
            'max_cohens_d': cohens_d_max,
            'z_scores': z_scores.tolist()
        },

        # 阈值设置
        'threshold': {
            'type': threshold_type,
            'value': effect_size_threshold,
            'raw_threshold': threshold_in_raw_scale,
            'interpretation': f"差异>{threshold_in_raw_scale:.4f}视为实际重要"
        },

        # 实际异质性统计量
        'practical': {
            'Q_scott_adjusted': Q_scott_adjusted,
            'p_value_adjusted': p_value_adjusted,
            'practical_heterogeneity_ratio': practical_heterogeneity_ratio,
            'n_important_groups': int(n_important),
            'proportion_important': proportion_important,
            'max_practical_effect': max_practical_effect,
            'z_scores_adjusted': z_scores_adjusted.tolist(),
            'is_practically_important': is_practically_important.tolist()
        },

        # 效应量分析
        'effect_sizes': {
            'cohens_d_max': cohens_d_max,
            'effect_sizes_vs_pooled': effect_sizes_vs_pooled.tolist(),
            'mean_effect_size': np.mean(effect_sizes_vs_pooled),
            'median_effect_size': np.median(effect_sizes_vs_pooled)
        },

        # 综合评估
        'assessment': {
            'has_practical_importance': has_practical_importance,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'scott_method_applicable': not has_practical_importance or risk_level in ['低', '中低'],
            'interpretation_summary': f"{k}个组中，{n_important}个({proportion_important:.1%})有实际重要差异。最大效应量d={cohens_d_max:.3f}。"
        }
    }


def assess_heterogeneity_by_overlap(data_groups, prior_mean=0.0, prior_variance=1.0,
                                    likelihood_variance=None, confidence_level=0.95,
                                    min_overlap_threshold=0.5):
    """
    基于置信区间重叠度的异质性评估（专为少组大样本设计）

    核心思想：当组数少但样本量大时，统计检验过于敏感。
    我们关心的是：各组估计的不确定性区间是否有实质性重叠。

    参数:
    data_groups: 数据组列表（通常2-5组）
    prior_mean, prior_variance: 先验参数
    likelihood_variance: 似然方差（None则从数据估计）
    confidence_level: 置信/可信水平（默认0.95）
    min_overlap_threshold: 最小可接受重叠度（0-1）
        0.5: 至少50%重叠
        0.8: 至少80%重叠（更严格）

    返回:
    基于重叠度的异质性评估结果
    """
    import numpy as np
    from scipy import stats

    k = len(data_groups)
    if k < 2:
        raise ValueError("至少需要2个数据组")

    # 1. 计算似然方差（如果未提供）
    if likelihood_variance is None:
        total_ss = 0
        total_df = 0
        for data in data_groups:
            if len(data) > 1:
                total_ss += np.sum((data - np.mean(data)) ** 2)
                total_df += len(data) - 1
        likelihood_variance = total_ss / total_df if total_df > 0 else 1.0

    # 2. 计算每组的后验统计量和置信区间
    posterior_means = []
    posterior_vars = []
    credible_intervals = []
    interval_widths = []

    prior_precision = 1.0 / prior_variance
    z_critical = stats.norm.ppf((1 + confidence_level) / 2)  # 例如1.96 for 95%

    for i, data in enumerate(data_groups):
        n_i = len(data)
        if n_i == 0:
            raise ValueError(f"第{i}组数据为空")

        # 样本统计
        x_bar_i = np.mean(data)

        # 计算后验
        likelihood_precision = n_i / likelihood_variance
        posterior_precision = prior_precision + likelihood_precision
        posterior_var_i = 1.0 / posterior_precision

        posterior_mean_i = (prior_precision * prior_mean +
                            likelihood_precision * x_bar_i) / posterior_precision

        posterior_means.append(posterior_mean_i)
        posterior_vars.append(posterior_var_i)

        # 计算置信区间
        std_err = np.sqrt(posterior_var_i)
        margin = z_critical * std_err
        ci_lower = posterior_mean_i - margin
        ci_upper = posterior_mean_i + margin

        credible_intervals.append((ci_lower, ci_upper))
        interval_widths.append(ci_upper - ci_lower)

    # 转换为numpy数组
    mu_hat = np.array(posterior_means)
    ci_array = np.array(credible_intervals)  # shape: (k, 2)

    # 3. 计算所有两两比较的重叠度
    overlap_ratios = []
    overlap_details = []

    for i in range(k):
        for j in range(i + 1, k):
            ci_i = ci_array[i]
            ci_j = ci_array[j]

            # 计算重叠区间
            overlap_start = max(ci_i[0], ci_j[0])
            overlap_end = min(ci_i[1], ci_j[1])

            if overlap_end > overlap_start:
                # 有重叠
                overlap_length = overlap_end - overlap_start

                # 计算重叠比例（相对于较窄的区间）
                width_i = ci_i[1] - ci_i[0]
                width_j = ci_j[1] - ci_j[0]
                narrower_width = min(width_i, width_j)

                if narrower_width > 0:
                    overlap_ratio = overlap_length / narrower_width
                else:
                    overlap_ratio = 1.0  # 如果区间宽度为0（理论上不可能）
            else:
                # 无重叠
                overlap_length = 0
                overlap_ratio = 0

            overlap_ratios.append(overlap_ratio)
            overlap_details.append({
                'group_pair': (i, j),
                'mean_diff': abs(mu_hat[i] - mu_hat[j]),
                'overlap_length': overlap_length,
                'overlap_ratio': overlap_ratio,
                'ci_i': ci_i.tolist(),
                'ci_j': ci_j.tolist()
            })

    # 4. 计算汇总统计量
    n_comparisons = len(overlap_ratios)

    if n_comparisons > 0:
        min_overlap = np.min(overlap_ratios)
        max_overlap = np.max(overlap_ratios)
        mean_overlap = np.mean(overlap_ratios)
        median_overlap = np.median(overlap_ratios)

        # 计算"充分重叠"的比例（超过阈值）
        sufficient_overlap_count = sum(ratio >= min_overlap_threshold for ratio in overlap_ratios)
        sufficient_overlap_prop = sufficient_overlap_count / n_comparisons
    else:
        min_overlap = max_overlap = mean_overlap = median_overlap = 1.0
        sufficient_overlap_prop = 1.0

    # 5. 基于重叠度的异质性判断
    # 核心逻辑：如果所有比较都有足够重叠，则认为无异质性
    all_sufficient_overlap = sufficient_overlap_prop == 1.0

    if all_sufficient_overlap:
        heterogeneity_level = "无实际异质性"
        risk_level = "低"
        recommendation = "✅ 所有组间置信区间充分重叠，Scott方法适用"
    elif sufficient_overlap_prop >= 0.8:
        heterogeneity_level = "轻微异质性"
        risk_level = "中低"
        recommendation = "⚠️  大部分组间重叠良好，Scott方法基本适用"
    elif sufficient_overlap_prop >= 0.5:
        heterogeneity_level = "中等异质性"
        risk_level = "中"
        recommendation = "⚠️  约半数组间重叠不足，需谨慎使用Scott方法"
    else:
        heterogeneity_level = "严重异质性"
        risk_level = "高"
        recommendation = "❌ 多数组间重叠不足，不建议使用简单Scott方法"

    # 6. 与传统Scott方法比较（作为参考）
    weights = 1.0 / np.array(posterior_vars)
    mu_pooled = np.sum(weights * mu_hat) / np.sum(weights)
    z_scores = (mu_hat - mu_pooled) / np.sqrt(posterior_vars)
    Q_scott = np.sum(z_scores ** 2)
    p_value_scott = 1 - stats.chi2.cdf(Q_scott, k)

    # 统计显著但可能实际不重要的情况
    is_statistically_significant = p_value_scott < 0.05
    has_practical_heterogeneity = not all_sufficient_overlap

    # 7. 计算效应量（Cohen's d）
    pooled_sd = np.sqrt(likelihood_variance)
    max_mean_diff = np.max(mu_hat) - np.min(mu_hat)
    cohens_d = max_mean_diff / pooled_sd if pooled_sd > 0 else 0

    return {
        # 基本信息
        'num_groups': k,
        'confidence_level': confidence_level,
        'min_overlap_threshold': min_overlap_threshold,

        # 后验统计量
        'posterior_means': mu_hat.tolist(),
        'posterior_vars': posterior_vars,
        'credible_intervals': ci_array.tolist(),
        'interval_widths': interval_widths,

        # 重叠度分析
        'overlap_summary': {
            'n_comparisons': n_comparisons,
            'min_overlap': min_overlap,
            'max_overlap': max_overlap,
            'mean_overlap': mean_overlap,
            'median_overlap': median_overlap,
            'sufficient_overlap_count': int(sufficient_overlap_count),
            'sufficient_overlap_prop': sufficient_overlap_prop,
            'all_sufficient_overlap': all_sufficient_overlap
        },
        'overlap_details': overlap_details,

        # 异质性判断
        'heterogeneity_assessment': {
            'heterogeneity_level': heterogeneity_level,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'all_sufficient_overlap': all_sufficient_overlap,
            'scott_method_applicable': all_sufficient_overlap or sufficient_overlap_prop >= 0.8
        },


        # 效应量
        'effect_size': {
            'pooled_sd': pooled_sd,
            'max_mean_diff': max_mean_diff,
            'cohens_d': cohens_d,
            'interpretation': f"最大标准化差异d={cohens_d:.3f}"
        },

        # 综合诊断
        'diagnosis': {
            'scenario': "少组大样本" if np.mean([len(g) for g in data_groups]) > 100 else "常规",
            'key_insight': "统计显著性与实际重要性可能不一致",
            'final_verdict': "基于重叠度方法" if not has_practical_heterogeneity else "需要进一步分析"
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
    data_A = np.array([
        np.random.normal(1000, 1, 10000),
        np.random.normal(999.9, 1, 10000),
        #np.random.normal(998, 1, 100)
    ])

    results_A = calculate_posterior_heterogeneity(
        data_A, prior_mean, prior_variance, likelihood_variance
    )
    results_A1 = assess_heterogeneity_by_overlap(data_A)
    print(results_A1['heterogeneity_assessment'])

    print(f"数据均值: {results_A['data_means']}")
    print(f"后验均值: {[f'{x:.4f}' for x in results_A['posterior_means']]}")
    print(f"后验异质性 I² = {results_A['scott']['scott_I2']:.2f}% Ex_heter = {results_A['scott']['excess_heterogeneity']:.2f}%")
    print(f"Q = {results_A['scott']['Q']:.4f}, p = {results_A['scott']['p_value']:.4f}")

    # 情况B：组1在-0.5附近，组2在0.5附近
    print("\n情况B：组1~N(-0.5,1), 组2~N(0.5,1)")
    print("-" * 40)

    data_B = np.array([
        np.random.normal(-0.05, 1, 1000),
        np.random.normal(0.05, 1, 1000)
    ])

    results_B = calculate_posterior_heterogeneity(
        data_B, prior_mean, prior_variance, likelihood_variance
    )

    print(f"数据均值: {results_B['data_means']}")
    print(f"后验均值: {[f'{x:.4f}' for x in results_B['posterior_means']]}")
    print(f"后验异质性 I² = {results_B['scott']['scott_I2']:.2f}% Ex_heter = {results_B['scott']['excess_heterogeneity']:.2f}%")
    print(f"Q = {results_B['scott']['Q']:.4f}, p = {results_B['scott']['p_value']:.4f}")



# 运行测试
if __name__ == "__main__":
    compare_heterogeneity_with_prior(None)