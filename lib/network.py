import argparse
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from sklearn.neighbors import NearestNeighbors

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


# def knn(x, k):
#     """
#     KNN's application
#     :param x: (batch_size, num_point, in_f), a tensor on CUDA
#     :param k: num of sampling points
#     :return: (batch_size, num_point, k, in_f), a tensor on CUDA
#     """
#     batch_size, num_point, in_f = x.size()
#     neigh = [NearestNeighbors(n_neighbors=k) for _ in range(batch_size)]
#     # Create new_x on the same device as x, assuming x is on CUDA
#     new_x = torch.zeros(batch_size, num_point, k, in_f, device=x.device)
#     for b in range(batch_size):
#         # Convert x[b] to CPU and numpy for sklearn compatibility
#         x_cpu_np = x[b].cpu().detach().numpy()
#         neigh[b].fit(x_cpu_np)
#         z = torch.zeros(num_point, k, in_f, device=x.device)  # Create z on CUDA
#         for i in range(num_point):
#             _, index = neigh[b].kneighbors(x_cpu_np[i].reshape(1, -1))
#             for t, j in enumerate(index[0]):
#                 z[i][t] = x[b][j]
#         new_x[b] = z
#     return new_x

def knn(x, k):
    """
    knn 's application
    :param x: (batch_size * num_point * in_f)
    :param k: num of sampling point
    :return: (batch_size * num_point * k * in_f)
    """
    batch_size, num_point, in_f = x.size()
    neigh = [NearestNeighbors(n_neighbors=k) for _ in range(batch_size)]
    new_x = torch.zeros(batch_size, num_point, k, in_f, device=x.device)
    for b in range(batch_size):
        x_cpu_np = x[b].cpu().detach().numpy()
        neigh[b].fit(x_cpu_np)
        z = torch.zeros(num_point, k, in_f, device=x.device)
        for i in range(num_point):
            _, index = neigh[b].kneighbors(x_cpu_np[i].reshape(1, -1))
            for t, j in enumerate(index[0]):
                z[i][t] = x[b][j]
        new_x[b] = z
    return new_x

def farthest_point_sample(x, samp_num):
    """
    FPS
    :param x: input [B,N,F]
    :param samp_num: num of sampled points
    :return:index of sampled points [B,samp_num]
    """
    device = x.device
    B, N, F = x.shape
    centroids = torch.zeros(B, samp_num, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(samp_num):
        # 更新第i个最远点
        centroids[:, i] = farthest
        # 取出这个最远点的xyz坐标
        centroid = x[batch_indices, farthest, :].view(B, 1, F)
        # 计算点集中的所有点到这个最远点的欧式距离
        # 等价于torch.sum((xyz - centroid) ** 2, 2)
        dist = torch.sum((x - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        # 取出每一行的最大值构成列向量，等价于torch.max(x,2)
        farthest = torch.max(distance, -1)[1]
    return centroids


def FPS(x, samp_num):
    """
    using fps algorithm
    :param x: input [B,N,F]
    :param samp_num: num of sampled points
    :return: sampled points [B,samp_num,F]
    """
    index = farthest_point_sample(x, samp_num)
    B, N, F = x.shape
    new_x = torch.zeros(B, samp_num, F, device='cuda')
    for i in range(B):
        for k, j in enumerate(index[i]):
            new_x[i][k] = x[i][j]
    return new_x

class SampGroup(nn.Module):
    def __init__(self, in_f, k, sam_num, out_f):
        """
        the model of Sampling&Grouping, after this you' ll get the result with [B,N,out_f]
        :param in_f: dim of input feature
        :param k: hyper parameter in knn
        :param sam_num: num of sampled points
        :param out_f: 3rd dim of output
        """
        super().__init__()
        self.k = k
        self.sam_num = sam_num
        self.out_f = out_f
        self.fc1 = nn.Linear(2 * in_f, self.out_f)
        self.fc2 = nn.Linear(self.out_f, self.out_f)
        self.bn = nn.BatchNorm2d(self.sam_num)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d((self.k, 1))

    def forward(self, x):
        B, p_num, N = x.shape
        if(self.sam_num < 500):
            sam_point = FPS(x, self.sam_num)  # sampling
        else:
            sam_point = x
        gro_point = knn(sam_point, self.k)  # grouping in sampled points
        sam_point = sam_point.repeat(1, self.k, 1, 1).view(B, self.sam_num, self.k, N)  # repeat to [B,sam_num,k,3]
        NKD = torch.sub(sam_point, gro_point)  # repeated sam_points subtract grouped points
        x = torch.cat((sam_point, NKD), dim=3)  # concat them to [B,sam_num,k,2*3]
        x = self.relu(self.bn((self.fc1(x))))  # first LBR
        x = self.relu(self.bn((self.fc2(x))))  # second LBR
        x = self.mp(x).view(B, self.sam_num, self.out_f)  # max pooling to [B,sam_num,out_f]
        return x

class InputEmbedding(nn.Module):
    def __init__(self, num_points, k1, k2, sam_num1, sam_num2):
        """
        input embedding, after this you' ll get the result with [B,N,128]
        :param num_points: num of points
        :param k1: first hyper parameter in grouping
        :param k2: second hyper parameter in grouping
        :param sam_num1: first hyper parameter in sampling
        :param sam_num2: second hyper parameter in sampling
        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.sam_num1 = sam_num1
        self.sam_num2 = sam_num2
        self.fc1 = nn.Linear(3, 64)
        self.bn = nn.BatchNorm1d(num_points)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.SG1 = SampGroup(64, self.k1, self.sam_num1, 64)
        self.SG2 = SampGroup(64, self.k2, self.sam_num2, 128)

    def forward(self, x):
        x = self.relu(self.bn((self.fc1(x))))
        x = self.relu(self.bn((self.fc2(x))))
        x = self.SG1(x)
        x = self.SG2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_f, dim_k=None, dim_v=None, transform='SS'):
        """
        Self Attention Mechanism
        :param in_f: dim of input feature
        :param dim_k: dim of query,key vector(default in_f)
        :param dim_v: dim of value vector,and also 3th dim of output(default in_f)
        :param transform: SS(default) means Scale + SoftMax,SL means SoftMax+L1Norm
        """
        super().__init__()
        self.dim_k = dim_k if dim_k else in_f
        self.dim_v = dim_v if dim_v else in_f
        self.transform = transform
        self.Q = nn.Linear(in_f, self.dim_k)
        self.K = nn.Linear(in_f, self.dim_k)
        self.V = nn.Linear(in_f, self.dim_v)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        B, _, _ = x.shape
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        if self.transform == 'SS':
            att_score = self.sm(torch.divide(torch.matmul(Q, K.permute(0, 2, 1)), math.sqrt(self.dim_k)))
        elif self.transform == 'SL':
            QK = torch.matmul(Q, K.permute(0, 2, 1))
            att_score = self.sm(QK)/QK.sum(dim=2).view(B, -1, 1)
        else:
            att_score = None
        Z = torch.matmul(att_score, V)
        return Z

class OffsetAttention(nn.Module):
    def __init__(self, num_points, in_f, dim_k=None, dim_v=None):
        """
        Offset-Attention
        :param num_points: num of points
        :param in_f: dim of input feature
        :param dim_k: dim of query,key vector(default in_f)
        :param dim_v: dim of value vector,and also 3th dim of output(default in_f)
        """
        super().__init__()
        self.dim_k = dim_k if dim_k else in_f
        self.dim_v = dim_v if dim_v else in_f
        self.sa = SelfAttention(in_f, self.dim_k, self.dim_v, 'SL')
        self.fc = nn.Linear(self.dim_v, self.dim_v)
        self.bn = nn.BatchNorm1d(num_points)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        atte_score = self.sa(x)
        x = self.relu(self.bn(self.fc(atte_score.sub(x)))).add(x)
        x = x.transpose(2, 1).contiguous()
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points, sam_num2):
        super(PoseNetFeat, self).__init__()
        # self.conv1 = torch.nn.Conv1d(32, 64, 1)
        # self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.offset_sa1 = OffsetAttention(sam_num2, 128)
        self.offset_sa2 = OffsetAttention(sam_num2, 128)
        self.offset_sa3 = OffsetAttention(sam_num2, 256)
        self.offset_sa4 = OffsetAttention(sam_num2, 512)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, emb):
        # x = F.relu(self.conv1(x))
        x = self.offset_sa1(x)

        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        # x = F.relu(self.conv2(x))
        x = self.offset_sa2(x)

        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        # x = F.relu(self.conv5(pointfeat_2))
        # x = F.relu(self.conv6(x))

        x = self.offset_sa3(pointfeat_2)
        x = F.relu(self.conv5(x))
        x = self.offset_sa4(x)
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj, sam_num2):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points, sam_num2)
        self.ec = InputEmbedding(num_points, k1=20, k2=20, sam_num1=num_points, sam_num2=sam_num2)
        self.fc = nn.Linear(128 * 4, 1024)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d((sam_num2, 1))
        self.bm = nn.BatchNorm1d(sam_num2)
        
        self.conv1_r = torch.nn.Conv1d(1472, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1472, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1472, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        x = self.ec(x)
        x = x.transpose(2, 1).contiguous()

        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        return out_rx, out_tx, out_cx, emb.detach()
 


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points, sam_num2):
        super(PoseRefineNetFeat, self).__init__()
        # self.conv1 = torch.nn.Conv1d(32, 64, 1)
        # self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(448, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.offset_sa1 = OffsetAttention(sam_num2, 128)
        self.offset_sa2 = OffsetAttention(sam_num2, 128)
        self.offset_sa3 = OffsetAttention(sam_num2, 448)
        self.offset_sa4 = OffsetAttention(sam_num2, 512)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        # x = F.relu(self.conv1(x))
        x = self.offset_sa1(x)
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        # x = F.relu(self.conv2(x))
        x = self.offset_sa2(x)
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = self.offset_sa3(pointfeat_3)
        x = F.relu(self.conv5(x))
        x = self.offset_sa4(x)
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj, sam_num2):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points, sam_num2)
        self.ec = InputEmbedding(num_points, k1=20, k2=20, sam_num1=num_points, sam_num2=sam_num2)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        x = self.ec(x)
        x = x.transpose(2, 1).contiguous()
        
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
