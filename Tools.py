import numpy as np
def shuffle_and_redistribute(data_groups):
    """合并数据，打乱后按照原始数据大小重新分组"""
    # 合并所有数据
    combined_data = np.concatenate(data_groups)

    # 打乱数据
    np.random.shuffle(combined_data)

    # 获取原始各组的大小
    original_sizes = [len(group) for group in data_groups]
    n_groups = len(data_groups)

    # 按照原始大小重新分配数据
    reshuffled_groups = []
    current_idx = 0

    for i in range(n_groups):
        size = original_sizes[i]
        reshuffled_groups.append(combined_data[current_idx:current_idx + size])
        current_idx += size

    # 验证分配正确
    reshuffled_sizes = [len(group) for group in reshuffled_groups]
    print(f"原始数据大小: {original_sizes}")
    print(f"重新分配后大小: {reshuffled_sizes}")

    return reshuffled_groups


def generate_heterogeneous_data(n_groups, mean, std, size):
    """生成3组具有异质性的测试数据"""

    data_groups = []
    for i in range(n_groups):
        data_groups.append(np.random.normal(loc=mean[i], scale=std[i], size=size[i]))

    return data_groups