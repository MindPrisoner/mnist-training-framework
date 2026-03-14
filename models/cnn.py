# import torch
# import torch.nn as nn
# import torch.nn.functional as F

#模型变化的历程
#1.简单的CNN
#2.LeNet架构
#3.加入了BatchNorm    解决  梯度消失   训练不稳定   收敛慢的问题   使得   稳定训练   加快收敛    允许更大学习率
#4.加入了DropOut      解决  防止过拟合   提升泛化能力




# class SimpleCNN(nn.Module):
#
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#
#         self.pool = nn.MaxPool2d(2)
#
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#
#         x = self.conv1(x)
#         x = F.relu(x)
#
#         x = self.conv2(x)
#         x = F.relu(x)
#
#         x = self.pool(x)
#
#         x = torch.flatten(x, 1)
#
#         x = self.fc1(x)
#         x = F.relu(x)
#
#         x = self.fc2(x)
#
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.pool = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x



#  加入了BatchNorm
# class LeNetBN(nn.Module):
#
#     def __init__(self):
#         super(LeNetBN, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.bn1 = nn.BatchNorm2d(6)
#
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.bn2 = nn.BatchNorm2d(16)
#
#         self.pool = nn.AvgPool2d(2)
#
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#
#         x = torch.flatten(x, 1)
#
#         x = F.relu(self.fc1(x))
#
#         x = F.relu(self.fc2(x))
#
#         x = self.fc3(x)
#
#         return x

#   加入了DropOut
# class LeNetBNDropout(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.bn1 = nn.BatchNorm2d(6)
#
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.bn2 = nn.BatchNorm2d(16)
#
#         self.pool = nn.AvgPool2d(2)
#
#         self.dropout = nn.Dropout(0.5)
#
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#
#         x = torch.flatten(x, 1)
#
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#
#         x = self.fc3(x)
#
#         return x

