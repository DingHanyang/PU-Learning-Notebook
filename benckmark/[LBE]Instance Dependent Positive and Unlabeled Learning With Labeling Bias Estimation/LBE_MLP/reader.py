import scipy.io as scio
import torch as t
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np

import os
import ssl
import torch
import pandas as pd
import urllib.request
from ucimlrepo import fetch_ucirepo

os.environ['http_proxy'] = '127.0.0.1:1066'
os.environ['https_proxy'] = '127.0.0.1:1066'

ssl._create_default_https_context = ssl._create_unverified_context


def _preprocess_data(uci_id):
    # fetch dataset
    dataset = fetch_ucirepo(id=uci_id)

    # data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets

    # 检测并删除缺失值
    combined = pd.concat([X, y], axis=1)
    combined_cleaned = combined.dropna()

    # 分离 features 和 targets
    X_cleaned = combined_cleaned.iloc[:, :-1]
    y_cleaned = combined_cleaned.iloc[:, -1]

    # 转换为 PyTorch 张量
    features = torch.Tensor(X_cleaned.to_numpy())
    targets = torch.Tensor(y_cleaned.to_numpy()).flatten()

    return features, targets


def get_uci_dataset_by_id(uci_id):

    match int(uci_id):
        case 15:
            # Breast
            features, targets = _preprocess_data(uci_id)
            # 转换标签为二分类
            targets = torch.where(targets != 4, torch.tensor(0.), torch.tensor(1.)) # 4为恶性 2为良性 改为1,0
            print(f'{features.shape}, p:{torch.sum(targets).item()}, n:{torch.sum(targets != 1).item()}')
            return features, targets
        case 143:
            # Australian
            features, targets = _preprocess_data(uci_id)
            # 转换标签为二分类
            targets = torch.where(targets != 1, torch.tensor(0.), torch.tensor(1.)) # 1 为正类 2为负 改为1,0
            print(f'{features.shape}, p:{torch.sum(targets).item()}, n:{torch.sum(targets != 1).item()}')
            return features, targets
        case 267:
            # banknote
            features, targets = _preprocess_data(uci_id)
            # 转换标签为二分类
            targets = torch.where(targets != 1, torch.tensor(0.), torch.tensor(1.)) # 1 为正类 2为负 改为1,0
            print(f'{features.shape}, p:{torch.sum(targets).item()}, n:{torch.sum(targets != 1).item()}')
            return features, targets
        case 327:
            # Phishing
            features, targets = _preprocess_data(uci_id)
            # 转换标签为二分类
            targets = torch.where(targets != 1, torch.tensor(0.), torch.tensor(1.)) # 1 为正类 2为负 改为1,0
            print(f'{features.shape}, p:{torch.sum(targets).item()}, n:{torch.sum(targets != 1).item()}')
            return features, targets




def reader_breast():
    data = scio.loadmat('datasets/breast.mat')['breast'].toarray()
    features, targets = t.Tensor(data[:, :-1]), t.Tensor(data[:, -1])
    targets = t.where(targets != 1, t.zeros_like(targets), targets)


    return get_uci_dataset_by_id(15)


def reader_australian():
    data = scio.loadmat('datasets/Australian.mat')
    features, targets = data['fea'], data['gnd']
    features, targets = t.Tensor(features.toarray()), t.Tensor(targets).squeeze()
    targets = t.where(targets != 1, t.zeros_like(targets), targets)
    return get_uci_dataset_by_id(267)


def syn(type, noise_rate, reader, ShrinkCoef=1, Power=1000):
    features, targets = reader()
    # features, targets = features.to('cuda'), targets.to('cuda')  # 这里什么都不做，直接采用cpu计算
    features_norm = (features - features.mean(0, keepdim=True)) / features.std(0, keepdim=True)  # normalize

    if type == 3:
        targets.squeeze_()
        targets_pu = t.where(t.rand_like(targets) < noise_rate, t.zeros_like(targets), targets)
    else:
        # model = nn.Linear(features_norm.size(1), 1).to('cuda')
        model = nn.Linear(features_norm.size(1), 1)
        optimizer = optim.Adam(model.parameters(), lr=.07)
        for i in range(30):
            optimizer.zero_grad()
            p = 1. / (1. + t.exp(-model(features_norm).squeeze()))
            loss = -(targets * p.log() + (1 - targets) * (1 - p).log()).mean()
            loss.backward()
            optimizer.step()
        temp = model(features_norm).squeeze().detach()
        print('\nsyn-logistic accu: %.2f' % ((temp > 0).float() == targets).float().mean())
        if type == 1:
            # 策略1
            p = (1. / (1. + t.exp(- temp))) ** Power
        else:
            # 策略2的变式
            p = (1.5 + ShrinkCoef - 1. / (1. + t.exp(- temp))) ** Power

        p = t.where(t.isnan(p), 1e-5, p)

        picknum = int(targets.sum() * (1 - noise_rate))
        targets_pu = mySampling(p * Power, picknum, targets, type == 1)

    return features, targets, targets_pu, model


def mySampling(py, n, targets, diu):
    py = py[targets == 1].cpu()
    iters = 1
    py /= py.sum()
    p_uni = t.ones_like(py)
    p_uni /= p_uni.sum()
    if diu:
        py = 0.9 * py + 0.1 * p_uni
    idx = t.Tensor(range(targets.size(0))).to(targets.device)[targets == 1].long()
    pu_targets = t.zeros_like(targets)
    # print(py)
    while pu_targets.sum() < n:
        tmp = np.random.choice(py.size(0), 1, True, py.numpy())
        pu_targets[idx[tmp]] = 1
        # write('syn-generate %d / %d iters %d' % (pu_targets.sum(), n, iters))
        iters += 1
    print()
    return pu_targets


def write(msg):
    sys.stdout.write(msg + '\b' * (len(msg) + 10))
