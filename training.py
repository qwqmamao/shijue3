# -*- coding: utf-8 -*-
"""
任务3：基础模型训练 + 通用训练/评估循环
=======================================
单个训练循环：记录 训练损失/训练准确率/验证损失/验证准确率，
保存训练日志(CSV)、损失&准确率曲线(PNG)、最优权重(PTP)。
供基础实验与控制变量对照实验复用。
"""

import os
import csv
import json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')          # 无界面绘图，直接存文件
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 让图中中文正常显示
plt.rcParams['axes.unicode_minus'] = False
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import CustomCNN
from dataset import get_dataloaders

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def train_one_epoch(model, loader, criterion, optimizer, device, use_aug=True,
                    scaler=None):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    autocast = torch.cuda.amp.autocast if device.type == 'cuda' else lambda: None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(x)
            loss = criterion(out, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return running_loss / total, correct / total


def save_curves(history, path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss'); ax[0].set_title('Loss曲线')
    ax[0].legend(); ax[0].grid(True)
    ax[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    ax[1].plot(epochs, history['val_acc'], 'r-o', label='Val Acc')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy'); ax[1].set_title('Accuracy曲线')
    ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run_training(cfg, name, device):
    """训练一个配置的模型并保存所有结果，返回最终 (val_acc, history)。

    支持断点续跑：每轮保存 last.pt（模型+优化器+历史），被中断后再次调用
    会从断点继续，直到全部轮次完成；完成后删除 last.pt 并写 summary.json。
    """
    os.makedirs(os.path.join(RESULTS, name), exist_ok=True)
    ckpt_path = os.path.join(RESULTS, name, 'last.pt')

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        cfg['batch_size'], num_workers=6)
    model = CustomCNN(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg.get('weight_decay', 0.0))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, best_state = 0.0, None
    start_epoch = 1
    eps = cfg['epochs']

    # ---- 断点续跑：若存在 last.pt 则恢复状态 ----
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        history = ckpt['history']
        best_val_acc, best_state = ckpt['best_val_acc'], ckpt['best_state']
        start_epoch = ckpt['epoch'] + 1
        print(f"[{name}] 发现断点，从 epoch {start_epoch} 续跑（已完成 {start_epoch - 1}/{eps}）")

    for epoch in range(start_epoch, eps + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer,
                                 device, scaler=scaler)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl)
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl);   history['val_acc'].append(va)
        print(f"[{name}] epoch {epoch:2d}/{eps} | "
              f"train_loss {tl:.4f} train_acc {ta:.4f} | "
              f"val_loss {vl:.4f} val_acc {va:.4f}")
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        # 每轮保存断点，进程被杀后可续跑
        torch.save({
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'history': history, 'best_val_acc': best_val_acc,
            'best_state': best_state, 'epoch': epoch, 'cfg': cfg,
        }, ckpt_path)

    # ---- 训练完成：写出最终产物并清理断点 ----
    torch.save({'state_dict': best_state, 'cfg': cfg, 'val_acc': best_val_acc},
               os.path.join(RESULTS, name, 'best_model.pt'))
    with open(os.path.join(RESULTS, name, 'history.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for i in range(len(history['train_loss'])):
            w.writerow([i + 1, history['train_loss'][i], history['train_acc'][i],
                        history['val_loss'][i], history['val_acc'][i]])
    with open(os.path.join(RESULTS, name, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    save_curves(history, os.path.join(RESULTS, name, 'curves.png'))
    summary = {**cfg, 'name': name, 'best_val_acc': round(best_val_acc, 4)}
    with open(os.path.join(RESULTS, name, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"[{name}] best val_acc = {best_val_acc:.4f}  >>> 已保存到 {RESULTS}/{name}\n")
    return best_val_acc, history


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = __import__('model').base_config()
    run_training(cfg, 'base', device)