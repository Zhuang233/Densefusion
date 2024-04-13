import torch
import _init_paths
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod


dataset = PoseDataset_linemod('train', 500, True, './datasets/linemod/Linemod_preprocessed', 0.03, False)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=10)

data = dataset.get_cloud_point_num_distribution()
data = np.nan_to_num(data, nan=0)

min_val = np.min(data)
max_val = np.max(data)
mean_val = np.mean(data)
median_val = np.median(data)
std_dev = np.std(data)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, color='skyblue', edgecolor='black')
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('lalala.png')  # 保存图像文件

print(min_val, max_val, mean_val, median_val, std_dev)