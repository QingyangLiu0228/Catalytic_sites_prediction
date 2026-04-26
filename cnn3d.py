#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cnn3d.py

3D-CNN architectures for voxelized catalytic site prediction.

Exports:
  - Model_1
  - CNN3D (alias of Model_1, used by train.py)
"""

import torch.nn as nn

# ============================================================================
# Main model
class Model_1(nn.Module):
    def __init__(self, dropout=0.5):
        super(Model_1, self).__init__()
        self.conv1 = nn.Conv3d(2560, 64, kernel_size=5, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU()
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # [50, 20, 20, 20, 2560] -> [50, 2560, 20, 20, 20]
        
        x = self.conv1(x)  # [50, 2560, 20, 20, 20] -> [50, 64, 8, 8, 8]
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)  # [50, 64, 8, 8, 8] -> [50, 128, 4, 4, 4]
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)  # [50, 128, 4, 4, 4] -> [50, 256, 2, 2, 2]
        x = self.batchnorm3(x)
        x = self.relu3(x)
        
        x = self.adaptive_pool(x)  # [50, 256, 2, 2, 2] -> [50, 256, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [50, 256, 1, 1, 1] -> [50, 256]
        x = self.fc(x)  # [50, 256] -> [50, 1]
        # x = self.sigmoid(x)
        return x


CNN3D = Model_1
