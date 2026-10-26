# -*- coding: utf-8 -*-
"""
手动搭建自定义卷积神经网络（任务2）
======================================
不调用 AlexNet / VGG / ResNet 等现成经典网络，
完全由手动态堆叠 卷积层(Conv2d) + 激活层(ReLU) + 池化层(MaxPool)
+ Dropout + 全连接层(Linear) 构成。

通过参数 cfg 动态决定网络结构，以便后续做"控制变量法"对照实验：
  kernel_size : 卷积核尺寸
  depth       : "卷积-激活-池化" 模块个数（网络深度）
  base_channels: 第一组卷积的通道数，后续每组翻倍（上限 256）
  dropout     : Dropout 丢弃概率（放在全连接层之间，0 表示不使用；卷积块间可选）
  pool_size   : 池化窗口尺寸（stride=pool_size, padding=(pool_size-1)//2，保持每次减半）
  fc_units    : 全连接隐藏层神经元数
  num_classes : 分类类别数（本实验为 8）
"""

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """手动堆叠的自定义卷积神经网络。"""

    def __init__(self, cfg):
        super().__init__()
        kernel_size = cfg['kernel_size']            # 卷积核尺寸
        depth = cfg['depth']                        # 卷积模块个数
        base_channels = cfg['base_channels']        # 基础通道数
        dropout = cfg['dropout']                    # Dropout 概率
        pool_size = cfg['pool_size']                # 池化窗口
        fc_units = cfg['fc_units']                  # 全连接神经元数
        num_classes = cfg['num_classes']            # 类别数
        self.input_size = cfg['img_size']           # 输入空间尺寸（正方形）

        self.features = nn.ModuleList()             # 卷积特征提取部分
        in_ch = 3                                   # 输入为 3 通道 RGB/OCT 图像
        spatial = self.input_size                   # 记录每次池化后的空间尺寸
        pad = (kernel_size - 1) // 2                # 为保证尺寸不变使用 SAME 填充

        for i in range(depth):
            out_ch = min(base_channels * (2 ** i), 256)   # 通道数逐层翻倍
            # ---- 卷积层：weight init 保持方差，增强梯度稳定性，这里用默认初始化即可 ----
            self.features.append(nn.Conv2d(in_ch, out_ch, kernel_size,
                                           stride=1, padding=pad))
            # ---- 激活层：ReLU ----
            self.features.append(nn.ReLU(inplace=True))
            # ---- 池化层：MaxPool，stride=pool_size 使每次空间尺寸减半 ----
            pool_pad = (pool_size - 1) // 2
            self.features.append(nn.MaxPool2d(kernel_size=pool_size,
                                              stride=pool_size, padding=pool_pad))
            # 记录前向传播后的特征图维度（用于维度核对）
            spatial = (spatial + 2 * pool_pad - pool_size) // pool_size + 1
            in_ch = out_ch

        # 展平后输入全连接的特征维度：C * H * W
        flat = in_ch * spatial * spatial
        self.flat_dim = flat

        # ---- 全连接分类头部：Linear + ReLU + Dropout + Linear ----
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return self.classifier(x)

    def describe(self):
        """打印每一层输出的特征图维度，用于核对维度流转是否正确。"""
        lines = []
        x = torch.empty(1, 3, self.input_size, self.input_size)
        lines.append(f"Input            : {tuple(x.shape)}")
        for i in range(0, len(self.features), 3):
            conv, relu, pool = self.features[i], self.features[i+1], self.features[i+2]
            x = relu(conv(x))
            lines.append(f"Conv{conv.kernel_size[0]}x{conv.kernel_size[0]} c={conv.out_channels}: {tuple(x.shape)}")
            x = pool(x)
            lines.append(f"Pool             : {tuple(x.shape)}")
        x = self.classifier(x)
        lines.append(f"Flatten/FC out   : {tuple(x.shape)}")
        return "\n".join(lines)


# 基础模型默认配置（其余对照实验仅更改其中一项）
def base_config():
    return {
        'kernel_size': 3,
        'depth': 5,
        'base_channels': 16,
        'dropout': 0.5,
        'pool_size': 2,
        'fc_units': 128,
        'img_size': 128,
        'num_classes': 8,
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 15,
        'weight_decay': 1e-4,
    }


if __name__ == '__main__':
    cfg = base_config()
    model = CustomCNN(cfg)
    print(model)
    print("\n---- 特征图维度核对 ----")
    print(model.describe())