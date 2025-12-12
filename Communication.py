import numpy as np


def normalize_subsets_by_swap_batch(data_list, swap_count):
    """
    批量交换版本：一次性找出多个最不合的点进行交换，效率更高。
    """
    swapped_data = [sublist[:] for sublist in data_list]
    n_subsets = len(swapped_data)

    if n_subsets < 2:
        return swapped_data

    # 找出差距最大的两个集合
    max_gap = -1
    best_pair = (0, 1)

    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            if len(swapped_data[i]) == 0 or len(swapped_data[j]) == 0:
                continue

            mean_i = np.mean(swapped_data[i])
            mean_j = np.mean(swapped_data[j])
            gap = abs(mean_i - mean_j)

            if gap > max_gap:
                max_gap = gap
                best_pair = (i, j)

    i, j = best_pair
    A = swapped_data[i]
    B = swapped_data[j]

    if len(A) == 0 or len(B) == 0:
        return swapped_data

    # 确定实际交换数量
    actual_swap_count = min(swap_count, len(A), len(B))

    if actual_swap_count == 0:
        return swapped_data

    # 批量找出最不合的点
    mean_B, std_B = np.mean(B), np.std(B)
    mean_A, std_A = np.mean(A), np.std(A)

    if std_B == 0:
        std_B = 1e-8
    if std_A == 0:
        std_A = 1e-8

    # 找出A中最不合B的多个点
    z_scores_A_in_B = [abs((x - mean_B) / std_B) for x in A]
    worst_A_indices = np.argsort(z_scores_A_in_B)[-actual_swap_count:][::-1]  # 从大到小排序

    # 找出B中最不合A的多个点
    z_scores_B_in_A = [abs((x - mean_A) / std_A) for x in B]
    worst_B_indices = np.argsort(z_scores_B_in_A)[-actual_swap_count:][::-1]

    # 批量交换
    points_from_A = [A[idx] for idx in worst_A_indices]
    points_from_B = [B[idx] for idx in worst_B_indices]

    # 执行交换
    for idx, point in zip(worst_A_indices, points_from_B):
        A[idx] = point

    for idx, point in zip(worst_B_indices, points_from_A):
        B[idx] = point

    return swapped_data


import matplotlib.pyplot as plt
from Tools import *
from Identification import calculate_posterior_heterogeneity

# 设置中文绘图
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成异质数据
data_subsets = generate_heterogeneous_data(4, [10, 0, -10, 1], std=[1, 1, 1, 1], size=[1000, 1000, 1000, 100])

print("原始数据:")
for i, subset in enumerate(data_subsets):
    print(f"集合{i + 1}: 均值: {np.mean(subset):.2f}, 标准差: {np.std(subset):.2f}")

# 记录Q值随交换次数的变化
swap_iterations = list(range(0, 101, 1))  # 从0到100次交换，每5次记录一次
Q_values = []

# 初始状态
heter0 = calculate_posterior_heterogeneity(data_subsets)
initial_Q = heter0['scott']['Q']
Q_values.append(initial_Q)
print(f"初始Q值: {initial_Q:.4f}")

# 进行多次交换并记录Q值
current_data = [subset[:] for subset in data_subsets]  # 创建副本

for iteration in swap_iterations[1:]:  # 跳过0次交换（已经是初始状态）
    # 每次交换10个点
    result = normalize_subsets_by_swap_batch(current_data, swap_count=10)

    # 更新当前数据为交换后的结果
    current_data = result

    # 计算异质性指标
    heter = calculate_posterior_heterogeneity(current_data)
    Q_value = heter['scott']['Q']
    Q_values.append(Q_value)

    if iteration % 20 == 0:  # 每20次交换打印一次进度
        print(f"交换{iteration}次后 Q值: {Q_value:.4f}")

# 绘制Q值随交换次数的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(swap_iterations, Q_values, 'b-o', linewidth=2, markersize=4)
plt.xlabel('交换次数', fontsize=12)
plt.ylabel('异质性指标 Q', fontsize=12)
plt.title('异质性指标 Q 随交换次数的变化', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=initial_Q, color='r', linestyle='--', alpha=0.7, label=f'初始Q值: {initial_Q:.4f}')
plt.legend()


plt.tight_layout()
plt.show()

# 输出最终结果
print(f"\n最终结果:")
print(f"初始Q值: {initial_Q:.4f}")
print(f"最终Q值: {Q_values[-1]:.4f}")

# 显示最终各集合的统计信息
print(f"\n最终各集合统计:")
for i, subset in enumerate(current_data):
    print(f"集合{i + 1}: 均值: {np.mean(subset):.2f}, 标准差: {np.std(subset):.2f}")