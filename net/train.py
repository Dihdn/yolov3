import torch
import torch.nn as nn
from dataset import *
from yolov3 import *
from config import *
from torch.utils.data import DataLoader

# 训练设备
device = torch.device('gpu:0' if torch.cuda.is_available() else 'cpu')

# 加载数据集
dataload = DataLoader(Yolo_dataset(), shuffle=True, batch_size=32)

# 加载网络
net = YoLov3net().to(device)

# 损失函数
cross_loss = nn.CrossEntropyLoss()  # 交叉熵损失
huber_loss = nn.HuberLoss()  # 平滑损失


# 优化器
optimizer = torch.optim.Adam(lr = LR)

for epoch in range(epochs):
    for img, label_13, label_26, label_52 in dataload:
        img = img.to(device)
        label_13 = label_13.to(device)
        label_26 = label_26.to(device)
        label_52 = label_52.to(device)
        pred_13, pred_26, pred_52 = net(img)


