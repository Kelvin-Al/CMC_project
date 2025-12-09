import numpy as np


class BaseMCMC:
    """普通MCMC实现"""

    def __init__(self, target_dist, proposal_std=1.0):
        self.target = target_dist
        self.proposal_std = proposal_std
        self.samples = []
        self.acceptance_rate = 0.0

    def metropolis_step(self, current):
        candidate = np.random.normal(current, self.proposal_std)
        acceptance_ratio = self.target(candidate) / self.target(current)

        if np.random.rand() < acceptance_ratio:
            return candidate, True
        else:
            return current, False

    def run_sampling(self, n_samples, initial_state, burn_in=0.2):
        current = initial_state
        accepted = 0
        total_iters = n_samples + int(n_samples * burn_in)
        self.samples = []

        for i in range(total_iters):
            current, accepted_flag = self.metropolis_step(current)

            if i >= int(n_samples * burn_in):
                self.samples.append(current)
                if accepted_flag:
                    accepted += 1

        self.acceptance_rate = accepted / n_samples
        return np.array(self.samples)


class ScottConsensusMCMC:
    """Scott加权平均共识MCMC (单进程版本)"""

    def __init__(self, data, num_subsets=4, proposal_std=1.0):
        self.data = data
        self.num_subsets = num_subsets
        self.proposal_std = proposal_std
        self.subsets = np.array_split(data, num_subsets)
        self.subposterior_samples = []
        self.subset_stats = []

    def subset_posterior(self, theta, subset_idx, prior_mean=0, prior_var=10):
        """计算子集后验分布"""
        X_k = self.subsets[subset_idx]
        n_k = len(X_k)
        data_var = 1

        # 共轭正态分布更新
        posterior_precision = 1 / prior_var + n_k / data_var
        posterior_var = 1 / posterior_precision
        posterior_mean = (prior_mean / prior_var + np.sum(X_k) / data_var) / posterior_precision

        return np.exp(-0.5 * (theta - posterior_mean) ** 2 / posterior_var)

    def run_subset_mcmc(self, n_samples_per_subset=1000):
        """在子集上顺序运行MCMC"""
        print("开始子集MCMC采样...")

        for subset_idx in range(self.num_subsets):
            print(f"正在采样子集 {subset_idx + 1}/{self.num_subsets}")

            def target(theta):
                return self.subset_posterior(theta, subset_idx)

            base_mcmc = BaseMCMC(target, self.proposal_std)
            samples = base_mcmc.run_sampling(n_samples_per_subset, 0.0)

            # 计算子后验统计量
            subset_mean = np.mean(samples)
            subset_var = np.var(samples)
            subset_precision = 1 / subset_var

            self.subset_stats.append({
                'samples': samples,
                'mean': subset_mean,
                'variance': subset_var,
                'precision': subset_precision,
                'acceptance_rate': base_mcmc.acceptance_rate
            })

        self.subposterior_samples = [stat['samples'] for stat in self.subset_stats]
        print("子集采样完成！")
        return self.subposterior_samples

    def form_consensus(self, n_consensus=1000):
        """Scott加权平均共识"""
        if not self.subset_stats:
            raise ValueError("请先运行子集MCMC")

        # 计算权重（基于精度）
        precisions = [stat['precision'] for stat in self.subset_stats]
        weights = np.array(precisions) / np.sum(precisions)

        print(f"子集权重: {weights}")

        consensus_samples = []
        for _ in range(n_consensus):
            # 从每个子后验抽取一个样本
            subsamples = []
            for i in range(self.num_subsets):
                random_idx = np.random.randint(len(self.subposterior_samples[i]))
                subsamples.append(self.subposterior_samples[i][random_idx])

            # Scott加权平均
            weighted_avg = np.average(subsamples, weights=weights)
            consensus_samples.append(weighted_avg)

        return np.array(consensus_samples)

    def get_subset_info(self):
        """获取子集信息"""
        info = []
        for i, stat in enumerate(self.subset_stats):
            info.append({
                'subset': i + 1,
                'data_size': len(self.subsets[i]),
                'mean': stat['mean'],
                'variance': stat['variance'],
                'precision': stat['precision'],
                'acceptance_rate': stat['acceptance_rate']
            })
        return info