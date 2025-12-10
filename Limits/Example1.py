import numpy as np
from Models.CMC_base import BaseMCMC, ConsensusMCMC
from plot import plot_comparison, plot_scott_results, plot_posterior_density
import matplotlib.pyplot as plt
import seaborn as sns
from Tools import shuffle_and_redistribute

def generate_heterogeneous_data():
    """生成3组具有异质性的测试数据"""
    np.random.seed(42)

    # 三组不同均值和方差的数据，模拟异质性
    data_groups = [
        np.random.normal(loc=-20.0, scale=1.0, size=1500),  # 低均值组
        np.random.normal(loc=0.0, scale=1.0, size=500),  # 中均值组
        np.random.normal(loc=20.0, scale=1.0, size=500)  # 高均值组
    ]

    print("异质性数据统计:")
    for i, data in enumerate(data_groups):
        print(f"组{i + 1}: 均值={data.mean():.3f}, 标准差={data.std():.3f}, 大小={len(data)}")

    return data_groups


def plot_comparison_results(consensus_samples, base_samples, reshuffled_samples,
                            true_mean,data_groups):
    """绘制三种方法的采样密度比较"""

    # 创建1行3列的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 绘制共识MCMC（异质分组）后验密度
    plot_posterior_density(consensus_samples, axes[0], 'blue', '共识MCMC(异质分组)', true_mean)
    axes[0].set_title('方法1: 共识MCMC - 异质分组')

    # 绘制普通MCMC后验密度
    plot_posterior_density(base_samples, axes[1], 'green', '普通MCMC', true_mean)
    axes[1].set_title('方法2: 普通MCMC - 合并数据')

    # 绘制共识MCMC（打乱分组）后验密度
    plot_posterior_density(reshuffled_samples, axes[2], 'orange', '共识MCMC(打乱分组)', true_mean)
    axes[2].set_title('方法3: 共识MCMC - 打乱分组')

    plt.tight_layout()
    plt.savefig('heterogeneity_density_comparison-Different_size.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print("\n后验分布统计信息:")
    print(f"方法1 - 共识MCMC(异质分组): 均值={np.mean(consensus_samples):.3f}, 标准差={np.std(consensus_samples):.3f}")
    print(f"方法2 - 普通MCMC(合并数据): 均值={np.mean(base_samples):.3f}, 标准差={np.std(base_samples):.3f}")
    print(
        f"方法3 - 共识MCMC(打乱分组): 均值={np.mean(reshuffled_samples):.3f}, 标准差={np.std(reshuffled_samples):.3f}")
    print(f"真实总体均值: {true_mean:.3f}")

    # 计算偏差
    bias1 = abs(np.mean(consensus_samples) - true_mean)
    bias2 = abs(np.mean(base_samples) - true_mean)
    bias3 = abs(np.mean(reshuffled_samples) - true_mean)

    print(f"\n估计偏差:")
    print(f"方法1偏差: {bias1:.3f}")
    print(f"方法2偏差: {bias2:.3f}")
    print(f"方法3偏差: {bias3:.3f}")

def main():
    """主函数：比较三种方法在异质性数据上的表现"""

    # 生成异质性数据
    print("=" * 50)
    print("生成异质性测试数据...")
    hetero_groups = generate_heterogeneous_data()
    full_data = np.concatenate(hetero_groups)
    true_overall_mean = full_data.mean()

    print(f"\n总体数据: 均值={true_overall_mean:.3f}, 大小={len(full_data)}")
    print("=" * 50)

    # 方法1: 直接使用共识MCMC（保持异质性分组）
    print("\n方法1: 共识MCMC（保持原始异质性分组）")
    print("-" * 40)
    consensus_mcmc = ConsensusMCMC(n_workers=3, n_samples=2000, n_burnin=1000)
    consensus_samples = consensus_mcmc.run_consensus(hetero_groups)

    consensus_mean = np.mean(consensus_samples)
    consensus_std = np.std(consensus_samples)
    print(f"共识MCMC结果: 均值={consensus_mean:.3f}, 标准差={consensus_std:.3f}")

    # 方法2: 合并所有数据后使用普通MCMC
    print("\n方法2: 普通MCMC（合并所有数据）")
    print("-" * 40)
    base_mcmc = BaseMCMC(n_samples=2000, n_burnin=1000)
    base_samples = base_mcmc.run(full_data)

    base_mean = np.mean(base_samples)
    base_std = np.std(base_samples)
    print(f"普通MCMC结果: 均值={base_mean:.3f}, 标准差={base_std:.3f}")

    # 方法3: 合并后打乱，重新分组使用共识MCMC
    print("\n方法3: 共识MCMC（打乱后重新分组）")
    print("-" * 40)
    reshuffled_groups = shuffle_and_redistribute(hetero_groups)

    # 显示重新分组后的统计信息
    print("重新分组后的数据统计:")
    for i, data in enumerate(reshuffled_groups):
        print(f"新组{i + 1}: 均值={data.mean():.3f}, 标准差={data.std():.3f}, 大小={len(data)}")

    reshuffled_consensus_mcmc = ConsensusMCMC(n_workers=3, n_samples=2000, n_burnin=1000)
    reshuffled_samples = reshuffled_consensus_mcmc.run_consensus(reshuffled_groups)

    reshuffled_mean = np.mean(reshuffled_samples)
    reshuffled_std = np.std(reshuffled_samples)
    print(f"重新分组共识MCMC结果: 均值={reshuffled_mean:.3f}, 标准差={reshuffled_std:.3f}")

    # 结果比较
    print("\n" + "=" * 50)
    print("结果比较:")
    print("=" * 50)
    print(f"真实总体均值: {true_overall_mean:.3f}")
    print(f"方法1 - 共识MCMC（异质分组）: {consensus_mean:.3f} (偏差: {abs(consensus_mean - true_overall_mean):.3f})")
    print(f"方法2 - 普通MCMC（合并数据）: {base_mean:.3f} (偏差: {abs(base_mean - true_overall_mean):.3f})")
    print(f"方法3 - 共识MCMC（打乱分组）: {reshuffled_mean:.3f} (偏差: {abs(reshuffled_mean - true_overall_mean):.3f})")

    # 可视化结果
    plot_comparison_results(consensus_samples, base_samples, reshuffled_samples,
                            true_overall_mean, hetero_groups)

    return consensus_samples, base_samples, reshuffled_samples




if __name__ == "__main__":
    # 运行比较实验
    consensus_samples, base_samples, reshuffled_samples = main()

    # 分析异质性问题的影响
    print("\n" + "=" * 60)
    print("异质性数据分析:")
    print("=" * 60)

    # 计算组间方差与组内方差的比值（异质性指标）
    hetero_groups = generate_heterogeneous_data()
    group_means = [group.mean() for group in hetero_groups]
    group_stds = [group.std() for group in hetero_groups]

    between_group_variance = np.var(group_means)  # 组间方差
    within_group_variance = np.mean([std ** 2 for std in group_stds])  # 组内方差均值

    heterogeneity_ratio = between_group_variance / within_group_variance
    print(f"组间方差: {between_group_variance:.4f}")
    print(f"组内方差均值: {within_group_variance:.4f}")
    print(f"异质性比率(组间/组内): {heterogeneity_ratio:.4f}")

    if heterogeneity_ratio > 0.1:
        print("⚠️  数据存在显著异质性，共识MCMC可能产生有偏估计")
        print("💡 建议: 打乱数据重新分组或使用加权共识方法")
    else:
        print("✅ 数据异质性较小，共识MCMC效果较好")