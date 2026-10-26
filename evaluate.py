# -*- coding: utf-8 -*-
"""
任务5：模型评估与错分样本分析
=============================
使用验证集上综合表现最优的模型在测试集上评估：
  - 计算测试集分类准确率与各类别准确率
  - 绘制混淆矩阵
  - 筛选被错误分类的样本并保存图像，便于观察错分原因
"""

import os
import json
import csv
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms

from model import CustomCNN
from dataset import get_dataloaders, CACHE_DIR

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'RetinalOCT_Dataset')
NORM = transforms.Normalize((0.5,) * 3, (0.5,) * 3)


def pick_best():
    """从 comparison.csv 选出验证集准确率最高的实验。"""
    with open(os.path.join(RESULTS, 'comparison.csv')) as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if r['val_acc'] not in ('', 'FAIL', None)]
    best = max(valid, key=lambda r: float(r['val_acc']))
    return best['experiment'], float(best['val_acc'])


def evaluate_test(best_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(os.path.join(RESULTS, best_name, 'best_model.pt'),
                      map_location=device)
    cfg, state = ckpt['cfg'], ckpt['state_dict']
    model = CustomCNN(cfg).to(device)
    model.load_state_dict(state)
    tr, vr, te, classes = get_dataloaders(cfg['batch_size'], num_workers=2)

    model.eval()
    all_pred, all_y = [], []
    correct = 0
    with torch.no_grad():
        for x, y in te:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu()
            all_pred.extend(pred.numpy())
            all_y.extend(y.numpy())
            correct += (pred == y).sum().item()
    acc = correct / len(all_y)
    print(f">>> 最优模型 {best_name} 在测试集上准确率 = {acc * 100:.2f}%\n")

    # 各类别准确率
    cm = confusion_matrix(all_y, all_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    with open(os.path.join(RESULTS, 'best_eval.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'test_accuracy'])
        for cls, a in zip(classes, per_class):
            w.writerow([cls, round(a, 4)])
    print("各类别准确率：")
    for cls, a in zip(classes, per_class):
        print(f"  {cls}: {a * 100:.2f}%")

    # 混淆矩阵图
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f'Confusion Matrix - {best_name} (Test Acc {acc*100:.2f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # 错分样本：加载缓存路径表，记录 路径+真实标签+预测标签
    with open(os.path.join(CACHE_DIR, 'test_paths.txt'), encoding='utf-8') as f:
        paths = f.read().split('\n')
    mis = []
    for idx, (pred, true) in enumerate(zip(all_pred, all_y)):
        if pred != true:
            mis.append((paths[idx], classes[true], classes[pred]))

    # 保存错分清单 JSON
    with open(os.path.join(RESULTS, 'misclassified.json'), 'w',
              encoding='utf-8') as f:
        json.dump([{'path': p, 'true': t, 'pred': pr} for p, t, pr in mis],
                  f, ensure_ascii=False, indent=2)

    # 保存前 12 个错分样本拼接图，便于观察
    _plot_misclassified(mis, classes, best_name)
    print(f"错分样本总数：{len(mis)} / {len(all_y)}")
    return acc, cm, mis


def _plot_misclassified(mis, classes, best_name, n=12):
    from PIL import Image
    k = min(n, len(mis))
    if k == 0:
        return
    cols = 3
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()
    for i in range(k):
        path, true, pred = mis[i]
        try:
            img = Image.open(path).copy()
        except FileNotFoundError:
            img = Image.new('RGB', (96, 96))
        axes[i].imshow(img)
        axes[i].set_title(f"T:{true}\nP:{pred}", fontsize=9)
        axes[i].axis('off')
    for j in range(k, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f'Misclassified samples - {best_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'misclassified.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    name, va = pick_best()
    print(f"从 results/comparison.csv 中选出最优模型：{name} (val_acc={va:.4f})")
    evaluate_test(name)