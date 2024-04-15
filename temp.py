# import torch

# def index_points(points, idx):
#     """
#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     device = points.device
#     B = points.shape[0] # 获取批次大小
#     view_shape = list(idx.shape) 
#     view_shape[1:] = [1] * (len(view_shape) - 1) # [1,1,1...]采样点数(S)个
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1 # [1, S]
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # [0~B]
#     new_points = points[batch_indices, idx, :]
#     return new_points



# points = torch.arange(24).view([2,3,4]).to("cuda")
# print(points)
# idx_list = [[1,2],
#             [2,0]]

# idx = torch.Tensor(idx_list).long()
# print(idx)


# result = index_points(points, idx)
# print(result)



# import numpy as np

# def sample_list_uniform_random(data, num_samples):
#     n = len(data)
#     if num_samples >= n:
#         return data

#     # 计算基本间隔
#     interval = n / num_samples

#     # 生成取样点索引
#     indices = [int(interval * i + np.random.uniform(-interval / 4, interval / 4)) for i in range(num_samples)]
#     # 确保索引在有效范围内
#     indices = [max(min(idx, n - 1), 0) for idx in indices]

#     # 获取取样数据
#     sampled_data = [data[idx] for idx in indices]

#     return sampled_data

# # 示例数据
# data_list = list(range(100))

# # 从列表中取10个样本
# sampled_data = sample_list_uniform_random(data_list, 10)
# print("取样数据:", sampled_data)

import numpy as np

class Sampler:
    def __init__(self, num):
        self.num = num

    def sample(self, choose):
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            c_mask = c_mask.nonzero()[0]  # 索引数组
            choose = choose[c_mask]      # 根据索引数组选择元素
        else:
            # 计算需要的重复次数以确保索引数组长度至少为self.num
            repeats = (self.num + len(choose) - 1) // len(choose)
            # 生成重复的索引数组并截取前self.num个元素
            c_mask = np.tile(np.arange(len(choose)), repeats)[:self.num]
            # 选择元素
            choose = choose[c_mask]

        return choose, c_mask

# 示例用法
sampler = Sampler(num=5)
choose = np.array([1, 2, 3, 4,5,6,7,8,9,10])
sampled_data, c_mask = sampler.sample(choose)
print("采样数据:", sampled_data)
print("采样掩码:", c_mask)
