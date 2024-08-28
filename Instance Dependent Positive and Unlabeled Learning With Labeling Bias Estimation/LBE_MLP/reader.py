import scipy.io as scio
import torch as t
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np


def reader_breast():
    data = scio.loadmat('datasets/breast.mat')['breast'].toarray()
    features, targets = t.Tensor(data[:, :-1]), t.Tensor(data[:, -1])
    targets = t.where(targets != 1, t.zeros_like(targets), targets)
    return features, targets


def reader_australian():
    data = scio.loadmat('datasets/Australian.mat')
    features, targets = data['fea'], data['gnd']
    features, targets = t.Tensor(features.toarray()), t.Tensor(targets).squeeze()
    targets = t.where(targets != 1, t.zeros_like(targets), targets)
    return features, targets


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
    #print(py)
    while pu_targets.sum() < n:
        tmp = np.random.choice(py.size(0), 1, True, py.numpy())
        pu_targets[idx[tmp]] = 1
        # write('syn-generate %d / %d iters %d' % (pu_targets.sum(), n, iters))
        iters += 1
    print()
    return pu_targets


def write(msg):
    sys.stdout.write(msg + '\b' * (len(msg) + 10))
