import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

def display_cloud(points_tensor):
    # 将数据从 CUDA 转移到 CPU，并转换为 NumPy 数组
    points = points_tensor[0].cpu().numpy()

    # 创建四个子图，每个子图从不同角度展示点云
    fig = plt.figure(figsize=(20, 15))
    angles = [(0, -90), (0, -45), (90, -45), (270, -45)]

    for i, angle in enumerate(angles, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')
        ax.view_init(elev=angle[1], azim=angle[0])
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'View angle {angle}')

    plt.suptitle('Point Cloud with Different Views')
    plt.savefig('cloud.png')  # 保存图像文件


def display_cloud2(points_tensor,a ,b, path):
    # 将数据从 CUDA 转移到 CPU，并转换为 NumPy 数组
    points = points_tensor[0].cpu().numpy()

    # 创建四个子图，每个子图从不同角度展示点云
    fig = plt.figure(figsize=(20, 15))

 
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')
    ax.view_init(elev=b, azim=a)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    plt.suptitle('Point Cloud with Different Views')
    plt.savefig(path+'cloud.png')  # 保存图像文件

def display_cloud_gif(points_tensor, path):
    # 将数据从 CUDA 转移到 CPU，并转换为 NumPy 数组
    points = points_tensor[0].cpu().numpy()

    # 创建帧并保存
    filenames = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for angle in range(-135, -45, 5):  # 每隔5度生成一个视角
        ax.view_init(angle, -90)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.7, s = 0.2)
        filename = f'frame_{angle}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        filenames.append(filename)
        ax.cla()  # 清除数据，准备下一个帧

    plt.close()

    # 步骤 3: 将所有帧组合成 GIF
    with imageio.get_writer(path+'point_cloud.gif', mode='I', duration=0.4) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # 如果你不再需要单独的帧文件，可以选择删除它们
    for filename in filenames:
        os.remove(filename)


def cloud_view_save(points_tensor, path):
    display_cloud_gif(points_tensor, path)
    display_cloud2(points_tensor,-90 ,-90, path)