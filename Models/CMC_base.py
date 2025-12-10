import numpy as np
from scipy import stats
from scipy.stats import norm

class BaseMCMC():
    """
    MCMC基类，定义基本的单链MCMC算法接口
    """

    def __init__(self, n_samples=1000, n_burnin=500):
        """
        初始化MCMC参数

        Parameters:
        -----------
        n_samples : int
            采样数量
        n_burnin : int
            预烧期数量
        """
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.samples = None
        self.acceptance_rate = 0.0
        self.proposal_std = 1.0

    def initialize_chain(self):
        """初始化链的起始位置"""
        return np.array([0.0])

    def proposal_distribution(self, current_state):
        """建议分布"""
        return current_state + np.random.normal(0, self.proposal_std, size = current_state.shape)


    def target_distribution(self, state, data=None):
        if data is None:
            return 0.0

        mu = state[0]
        # 似然函数：数据来自N(mu, 1)
        log_likelihood = np.sum(norm.logpdf(data, loc=mu, scale=1.0))
        # 先验：mu ~ N(0, 10)
        log_prior = norm.logpdf(mu, loc=0, scale=10.0)
        return log_likelihood + log_prior

    def run(self, data=None):
        """
        运行MCMC采样

        Parameters:
        -----------
        data : array-like, optional
            观测数据

        Returns:
        --------
        samples : ndarray
            采样结果
        """
        n_total = self.n_samples + self.n_burnin
        dim = len(self.initialize_chain())
        self.samples = np.zeros((n_total, dim))

        # 初始化
        current_state = self.initialize_chain()
        current_log_prob = self.target_distribution(current_state, data)

        accept_count = 0

        for i in range(n_total):
            # 从建议分布中采样
            proposed_state = self.proposal_distribution(current_state)
            proposed_log_prob = self.target_distribution(proposed_state, data)

            # 计算接受概率
            log_acceptance_ratio = proposed_log_prob - current_log_prob
            acceptance_prob = min(1, np.exp(log_acceptance_ratio))

            # 决定是否接受新状态
            if np.random.rand() < acceptance_prob:
                current_state = proposed_state
                current_log_prob = proposed_log_prob
                accept_count += 1

            self.samples[i] = current_state

        # 计算接受率
        self.acceptance_rate = accept_count / n_total

        # 去除预烧期
        return self.samples[self.n_burnin:]

    def get_posterior_mean(self):
        """获取后验均值"""
        if self.samples is None:
            raise ValueError("请先运行MCMC采样")
        return np.mean(self.samples[self.n_burnin:], axis=0)

    def get_posterior_std(self):
        """获取后验标准差"""
        if self.samples is None:
            raise ValueError("请先运行MCMC采样")
        return np.std(self.samples[self.n_burnin:], axis=0)


class ConsensusMCMC(BaseMCMC):
    """
    共识MCMC类，使用Scott的加权平均方法
    """

    def __init__(self, n_workers=4, n_samples=1000, n_burnin=500):
        """
        初始化共识MCMC参数

        Parameters:
        -----------
        n_workers : int
            工作节点数量（数据子集数量）
        n_samples : int
            每节点采样数量
        n_burnin : int
            每节点预烧期数量
        """
        super().__init__(n_samples, n_burnin)
        self.n_workers = n_workers
        self.subset_data = None
        self.worker_samples = None
        self.consensus_samples = None

    def set_subset_data(self, subset_data):
        """
        设置子集数据

        Parameters:
        -----------
        subset_data : list of array-like
            各工作节点的数据子集列表
        """
        if len(subset_data) != self.n_workers:
            raise ValueError(f"子集数据数量必须等于工作节点数量 {self.n_workers}")
        self.subset_data = subset_data

    def scott_weighted_average(self, worker_samples):
        """
        Scott的加权平均方法

        Parameters:
        -----------
        worker_samples : list of ndarray
            各工作节点的采样结果

        Returns:
        --------
        consensus_samples : ndarray
            共识采样结果
        """
        n_samples = worker_samples[0].shape[0]
        dim = worker_samples[0].shape[1]
        consensus_samples = np.zeros((n_samples, dim))

        for t in range(n_samples):
            # 收集当前时间步所有工作节点的样本
            current_samples = np.array([worker_samples[i][t] for i in range(self.n_workers)])

            # 计算加权平均（这里使用简单的算术平均作为示例）
            # 实际应用中可能需要根据各节点的精度或数据量进行加权
            consensus_samples[t] = np.mean(current_samples, axis=0)

        return consensus_samples

    def run_consensus(self, full_data=None):
        """
        运行共识MCMC

        Parameters:
        -----------
        full_data : array-like, optional
            完整数据集（如果提供，将自动划分为子集）

        Returns:
        --------
        consensus_samples : ndarray
            共识采样结果
        """
        if full_data is not None:
            self._split_data(full_data)

        if self.subset_data is None:
            raise ValueError("请先设置子集数据或提供完整数据")

        self.worker_samples = []

        # 在各数据子集上并行运行MCMC（这里用循环模拟并行）
        for i in range(self.n_workers):
            print(f"在工作节点 {i + 1} 上运行MCMC...")
            worker_mcmc = self._create_worker_mcmc()
            samples = worker_mcmc.run(self.subset_data[i])
            self.worker_samples.append(samples)

        # 应用Scott加权平均
        self.consensus_samples = self.scott_weighted_average(self.worker_samples)
        return self.consensus_samples

    def _split_data(self, full_data):
        """将完整数据划分为子集"""
        n_data = len(full_data)
        subset_size = n_data // self.n_workers
        self.subset_data = []

        for i in range(self.n_workers):
            start_idx = i * subset_size
            end_idx = start_idx + subset_size if i < self.n_workers - 1 else n_data
            self.subset_data.append(full_data[start_idx:end_idx])

    def _create_worker_mcmc(self):
        """创建工作节点专用的MCMC实例"""
        return BaseMCMC(n_samples=self.n_samples, n_burnin=self.n_burnin)

    # 重写基类方法以适应共识MCMC
    def run(self, data=None):
        """运行共识MCMC的便捷方法"""
        return self.run_consensus(data)

    def get_posterior_mean(self):
        """获取共识后验均值"""
        if self.consensus_samples is None:
            raise ValueError("请先运行共识MCMC采样")
        return np.mean(self.consensus_samples, axis=0)

    def get_posterior_std(self):
        """获取共识后验标准差"""
        if self.consensus_samples is None:
            raise ValueError("请先运行共识MCMC采样")
        return np.std(self.consensus_samples, axis=0)