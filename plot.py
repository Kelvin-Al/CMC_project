import matplotlib.pyplot as plt
import numpy as np

# 设置中文绘图
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_comparison(base_samples, consensus_samples, true_theta=None):
    """绘制普通MCMC和共识MCMC比较图"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(base_samples[:1000], alpha=0.7, label='普通MCMC')
    plt.plot(consensus_samples[:1000], alpha=0.7, label='共识MCMC')
    if true_theta is not None:
        plt.axhline(y=true_theta, color='red', linestyle='--', alpha=0.5, label='真实值')
    plt.title('MCMC样本链')
    plt.xlabel('迭代次数')
    plt.ylabel('参数值')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(base_samples, bins=50, density=True, alpha=0.6, label='普通MCMC')
    plt.hist(consensus_samples, bins=50, density=True, alpha=0.6, label='共识MCMC')
    if true_theta is not None:
        plt.axvline(x=true_theta, color='red', linestyle='--', label='真实值')
    plt.title('后验分布比较')
    plt.xlabel('参数值')
    plt.ylabel('概率密度')
    plt.legend()

    plt.subplot(1, 3, 3)
    # 收敛诊断
    sample_sizes = range(100, min(len(base_samples), len(consensus_samples)), 100)
    base_means = [np.mean(base_samples[:n]) for n in sample_sizes]
    consensus_means = [np.mean(consensus_samples[:n]) for n in sample_sizes]

    plt.plot(sample_sizes, base_means, 'o-', label='普通MCMC均值')
    plt.plot(sample_sizes, consensus_means, 's-', label='共识MCMC均值')
    if true_theta is not None:
        plt.axhline(y=true_theta, color='red', linestyle='--', label='真实值')
    plt.title('均值收敛')
    plt.xlabel('样本数量')
    plt.ylabel('参数均值')
    plt.legend()

    plt.tight_layout()
    return plt


def plot_scott_results(subset_samples, consensus_samples, subset_info, true_theta):
    """绘制Scott共识MCMC结果"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for i, samples in enumerate(subset_samples):
        plt.hist(samples, bins=50, density=True, alpha=0.6,
                 label=f'子集{i + 1} (n={subset_info[i]["data_size"]})')
    plt.axvline(x=true_theta, color='red', linestyle='--', label='真实参数')
    plt.title('子集后验分布')
    plt.xlabel('参数值')
    plt.ylabel('密度')
    plt.legend()

    plt.subplot(1, 3, 2)
    weights = [info['precision'] for info in subset_info]
    plt.bar(range(1, len(weights) + 1), weights)
    plt.title('子集权重（精度）')
    plt.xlabel('子集编号')
    plt.ylabel('权重')

    plt.subplot(1, 3, 3)
    plt.hist(consensus_samples, bins=50, density=True, alpha=0.6,
             label=f'共识后验 (均值={np.mean(consensus_samples):.3f})')
    plt.axvline(x=true_theta, color='red', linestyle='--',
                label=f'真实参数 ({true_theta})')
    plt.title('Scott加权共识结果')
    plt.xlabel('参数值')
    plt.ylabel('密度')
    plt.legend()

    plt.tight_layout()
    return plt