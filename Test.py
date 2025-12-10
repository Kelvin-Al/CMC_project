import numpy as np
from Models.CMC_base import BaseMCMC, ConsensusMCMC
from plot import plot_comparison, plot_scott_results

def main():
    """主函数"""
    # 生成测试数据
    np.random.seed(42)
    true_mu = 2.5
    data_sizes = [100, 200, 300, 400]
    subset_data = []

    # 生成4个不同大小的数据子集
    for size in data_sizes:
        data = np.random.normal(true_mu, 1, size)
        subset_data.append(data)

    # 合并所有数据
    full_data = np.concatenate(subset_data)
    print(f"子集大小: {data_sizes}")
    print(f"总数据大小: {len(full_data)}")

    # 1. 使用共识MCMC运行
    print("运行共识MCMC...")
    consensus_mcmc = ConsensusMCMC(n_workers=4, n_samples=2000, n_burnin=1000)
    consensus_samples = consensus_mcmc.run_consensus(full_data)

    # 2. 合并所有数据用普通MCMC运行
    print("运行普通MCMC...")
    base_mcmc = BaseMCMC(n_samples=20000, n_burnin=10000)
    base_samples = base_mcmc.run(full_data)

    # 输出结果比较
    print(f"\n真实参数值: {true_mu}")
    print(f"共识MCMC后验均值: {np.mean(consensus_samples):.4f}")
    print(f"普通MCMC后验均值: {np.mean(base_samples):.4f}")
    print(f"共识MCMC后验标准差: {np.std(consensus_samples):.4f}")
    print(f"普通MCMC后验标准差: {np.std(base_samples):.4f}")

    # 使用plot.py中的函数绘图
    plt1 = plot_comparison(base_samples, consensus_samples, true_mu)
    plt1.show()

    # 如果有plot_scott_results函数需要的额外信息
    subset_info = [
        {'data_size': 100, 'precision': 1 / 0.1},
        {'data_size': 200, 'precision': 1 / 0.08},
        {'data_size': 300, 'precision': 1 / 0.06},
        {'data_size': 400, 'precision': 1 / 0.05}
    ]

    plt2 = plot_scott_results(consensus_mcmc.worker_samples, consensus_samples, subset_info, true_mu)
    plt2.show()
if __name__ == '__main__':
    main()