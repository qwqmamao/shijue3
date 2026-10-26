# -*- coding: utf-8 -*-
"""
任务1：数据集加载与预处理
=========================
统计数据集基本信息（类别、样本总量、分辨率、通道数）；
完成图像缩放(to 128x128)、像素归一化(to [0,1]);
划分训练集 / 验证集 / 测试集（数据集已按目录划分好 train/val/test），
测试集不参与调参，仅在任务5最终评估使用。

性能优化：
  由于所有对照实验都会反复读图，这里将"解码+缩放"后的图像一次性缓存为
  uint8 张量到 experiment3/cache/*.npy，训练时直接读缓存并做轻量预处理
  (归一化、随机翻转)，避免每个 epoch 都重复解码大尺寸 JPEG 图像。
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RetinalOCT_Dataset'))
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
IMG_SIZE = 128


# ---------- 数据集基本信息统计 ----------
def dataset_summary():
    print("=" * 60)
    print("OCT 眼底疾病 8 分类数据集基本信息")
    print("=" * 60)
    summary = {}
    for split in ['train', 'val', 'test']:
        d = os.path.join(BASE_DIR, split)
        summary[split] = {}
        for cls in sorted(os.listdir(d)):
            p = os.path.join(d, cls)
            files = os.listdir(p)
            sz, mode = None, None
            if files:
                im = Image.open(os.path.join(p, files[0]))
                sz, mode = im.size, im.mode
            summary[split][cls] = (len(files), sz, mode)
            print(f"[{split}] {cls}: {len(files)} 样本, 示例尺寸={sz}, 通道={mode}")
    return summary


# ---------- 一次性缓存预处理（解码+缩放 -> uint8 张量） ----------
def _cache_split(split):
    """构建 split 的缓存（解码+RGB拉伸缩放 -> uint8 张量）。

    旧版方案：RGB 三通道直接 resize 成 128x128（Image.BILINEAR）。
    不去重、不转灰度、不保持长宽比。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'{split}.npy')
    label_file = os.path.join(CACHE_DIR, f'{split}_labels.npy')
    if os.path.exists(cache_file) and os.path.exists(label_file):
        return cache_file, label_file

    d = os.path.join(BASE_DIR, split)
    classes = sorted(os.listdir(d))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    imgs, labels, paths = [], [], []
    for cls in classes:
        for f in sorted(os.listdir(os.path.join(d, cls))):
            p = os.path.join(d, cls, f)
            img = Image.open(p).convert('RGB').resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            imgs.append(np.asarray(img).transpose(2, 0, 1))  # HWC->CHW, uint8
            labels.append(class_to_idx[cls])
            paths.append(p)
    arr = np.stack(imgs).astype(np.uint8)
    np.save(cache_file, arr)
    np.save(label_file, np.array(labels, dtype=np.int64))
    with open(os.path.join(CACHE_DIR, f'{split}_paths.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(paths))
    return cache_file, label_file


# ---------- 基于缓存的 Dataset ----------
class CachedOCTDataset(Dataset):
    def __init__(self, split, augment=False):
        self.augment = augment
        img_file, label_file = _cache_split(split)
        self.data = np.load(img_file, mmap_mode='r')    # 内存映射，避免占满内存
        self.labels = np.load(label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.data[idx]))  # uint8 CHW
        x = x.to(torch.float32) / 255.0                   # 归一化到 [0,1]
        x = (x - 0.5) / 0.5                               # 线性映射到 [-1,1]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])               # 随机水平翻转
        return x, int(self.labels[idx])


def get_dataloaders(batch_size=128, num_workers=4, test_partition='val'):
    train_ds = CachedOCTDataset('train', augment=True)
    val_ds = CachedOCTDataset('val')
    test_ds = CachedOCTDataset('test')
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=True)
    vr = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=True)
    te = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=True)
    classes = [c for c in sorted(os.listdir(os.path.join(BASE_DIR, 'train')))]
    return tr, vr, te, classes


if __name__ == '__main__':
    dataset_summary()
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    x, y = next(iter(train_loader))
    print("训练集样本总数：", len(train_loader.dataset))
    print("单个 batch 输入维度：", tuple(x.shape), " 标签维度：", tuple(y.shape),
          " 类别数：", len(classes))