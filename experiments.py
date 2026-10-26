# -*- coding: utf-8 -*-
"""
任务4：控制变量法超参数对照实验
===============================
以基础配置为基准，每次只修改一个变量做对照：
  1. 卷积核大小对照：kernel_size ∈ {3,5,7}
  2. 网络深度对照：  depth ∈ {3,5,6}
  3. Dropout 概率对照：dropout ∈ {0,0.3,0.5,0.7}
  4. 学习率对照：     learning_rate ∈ {0.01,0.001,0.0001}
  5. 池化窗口对照：   pool_size ∈ {2,3}
  6. 训练轮次对照：   epochs ∈ {10,15,20}
运行所有实验并把结果汇总到 results/comparison.csv，供报告制表。
"""

import os
import csv
import copy
import json
import torch

from model import base_config
from training import run_training

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def build_experiments():
    """生成所有对照实验配置：name, 说明, 修改哪一项。"""
    base = base_config()
    exps = []

    def add(exp_name, mutator):
        cfg = copy.deepcopy(base)
        mutator(cfg)
        exps.append((exp_name, cfg))

    # 卷积核大小对照（其余不变）
    for k in [5, 7]:
        add(f'kernel_{k}', lambda c, kk=k: c.update(kernel_size=kk))
    # 网络深度对照（浅层、相对深层）
    for d in [3, 6]:
        add(f'depth_{d}', lambda c, dd=d: c.update(depth=dd))
    # Dropout 概率对照
    for p in [0.0, 0.3, 0.7]:
        add(f'drop_{int(p*100):02d}', lambda c, pp=p: c.update(dropout=pp))
    # 学习率对照
    for lr in [0.01, 0.0001]:
        add(f'lr_{str(lr).replace(".","p")}', lambda c, ll=lr: c.update(learning_rate=ll))
    # 池化窗口对照
    add('pool_3', lambda c: c.update(pool_size=3))
    # 训练轮次对照
    for e in [10, 20]:
        add(f'epoch_{e}', lambda c, ee=e: c.update(epochs=ee))

    return exps


def main(budget_minutes=None):
    """按时间预算运行对照实验。

    budget_minutes: 可选，总运行时长上限（分钟）。后台作业可能被系统在
    约40分钟后终止，设置预算（如32分钟）可在被杀前干净收尾，下次重跑时
    通过 summary.json / last.pt 断点续跑，已完成的实验自动跳过。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备：", device)
    import time
    t0 = time.time()
    results = []

    all_cfgs = [('base', base_config())] + build_experiments()
    for name, cfg in all_cfgs:
        # 检查时间预算：超出则停止，交由下次续跑
        if budget_minutes and (time.time() - t0) > budget_minutes * 60:
            print(f"\n>>> 已达到时间预算 {budget_minutes} 分钟，提前收尾（剩余未跑实验下次续跑）")
            break

        summary_file = os.path.join(RESULTS, name, 'summary.json')
        ckpt_file = os.path.join(RESULTS, name, 'last.pt')
        if os.path.exists(summary_file):
            with open(summary_file, encoding='utf-8') as f:
                va = json.load(f).get('best_val_acc')
            results.append({'name': name, 'val_acc': va})
            print(f"[{name}] 已存在，跳过（val_acc={va}）\n")
            continue
        try:
            va, _ = run_training(cfg, name, device)
            results.append({'name': name, 'val_acc': va})
        except Exception as e:
            print(f"[{name}] 训练失败: {e}\n")
            results.append({'name': name, 'val_acc': None})

    write_comparison(results)
    valid = [r for r in results if r['val_acc'] is not None]
    best = max(valid, key=lambda r: r['val_acc']) if valid else None
    if best:
        print(f"\n>>> 综合最优：{best['name']}  val_acc = {best['val_acc']:.4f}  "
              f"(汇总见 results/comparison.csv)")
    return best


def write_comparison(results):
    rows = [['experiment', 'kernel_size', 'depth', 'dropout', 'pool_size',
             'learning_rate', 'epochs', 'val_acc']]
    cfgs = {'base': base_config()}
    for name, cfg in build_experiments():
        cfgs[name] = cfg
    for r in results:
        c = cfgs.get(r['name'], {})
        rows.append([r['name'], c.get('kernel_size'), c.get('depth'),
                     c.get('dropout'), c.get('pool_size'),
                     c.get('learning_rate'), c.get('epochs'),
                     r['val_acc'] if r['val_acc'] is not None else 'FAIL'])
    with open(os.path.join(RESULTS, 'comparison.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"已生成 results/comparison.csv（{len(rows) - 1} 条记录）")


if __name__ == '__main__':
    import sys
    # 通过命令行传预算（分钟），如：python experiments.py 32
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(budget_minutes=budget)