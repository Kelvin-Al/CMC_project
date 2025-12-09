import numpy as np
from Models.CMC_base import BaseMCMC, ScottConsensusMCMC
from plot import plot_comparison, plot_scott_results

def main():
    """主函数"""
    # 生成数据
    np.random.seed(42)
    true_theta = 2.0
    data = np.random.normal(true_theta, 1, 1000)

    print("=== Scott共识MCMC演示 (单进程版本) ===")

    # 1. 普通MCMC
    def target_distribution(x):
        return np.exp(-0.5 * (x-1)**2) + 0.5 * np.exp(-0.5 * (x+2)**2)

    base_mcmc = BaseMCMC(target_distribution)
    base_samples = base_mcmc.run_sampling(5000, 0.0)

    # 2. Scott共识MCMC
    scott_mcmc = ScottConsensusMCMC(data, num_subsets=4)
    subset_samples = scott_mcmc.run_subset_mcmc(1000)  # 减少样本数以加快速度
    consensus_samples = scott_mcmc.form_consensus(1000)
    subset_info = scott_mcmc.get_subset_info()

    # 打印结果
    print("\n子集信息:")
    for info in subset_info:
        print(f"子集{info['subset']}: 数据量={info['data_size']}, "
              f"均值={info['mean']:.3f}, 方差={info['variance']:.3f}, "
              f"权重={info['precision']:.3f}")

    print(f"\n普通MCMC接受率: {base_mcmc.acceptance_rate:.3f}")
    print(f"共识MCMC均值: {np.mean(consensus_samples):.3f}")
    print(f"真实参数: {true_theta:.3f}")

    # 绘图
    plot_comparison(base_samples, consensus_samples, true_theta).show()
    plot_scott_results(subset_samples, consensus_samples, subset_info, true_theta).show()

if __name__ == '__main__':
    main()