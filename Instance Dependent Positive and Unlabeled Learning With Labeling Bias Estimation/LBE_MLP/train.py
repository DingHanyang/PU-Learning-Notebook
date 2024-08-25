import torch as t
import torch.nn as nn

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from em import *

from reader import *

import sys

E_STEPS, M_STEPS = 20, 20

class MLP(nn.Module):
    def __init__(self, dim_feature):
        super().__init__()
        self.fc1 = nn.Linear(dim_feature, int(100))
        self.fc2 = nn.Linear(int(100), 1)
        
    def forward(self, inputs):
        x = self.fc1(inputs)
        output =  self.fc2(x.tanh())       
        return output

def pretrain(x_train, q_train, model_pyx, model_eta, lr, wd):
    q_train = q_train.view(-1,1).float()
    porportion_labeled = q_train.mean()
    optimizer = Adam(model_pyx.parameters(), lr = lr, weight_decay = wd)
    for iter_stp in range(100):
        optimizer.zero_grad()
        p = model_pyx(x_train).sigmoid()
        loss = -(q_train * (1-porportion_labeled) * p.log() + porportion_labeled * (1-q_train) * (1-p).log()).mean()
        
        loss.backward()
        optimizer.step()

    classification_expectation = model_pyx(x_train).sigmoid().detach()
    weights = q_train + (1-q_train) * classification_expectation
    optimizer = Adam(model_eta.parameters(), lr = lr, weight_decay = wd)
    for iter_stp in range(100):
        optimizer.zero_grad()
        p = model_eta(x_train).sigmoid()
        loss = -(q_train * (1-porportion_labeled) * p.log() + porportion_labeled * (1-q_train) * (1-p).log()).mean()
        loss.backward()
        optimizer.step()
    return model_pyx, model_eta

def em_train(x_train, q_train, x_test, y_test, lr, wd):
    dim_fea = x_train.size(1)
    model_pyx = MLP(dim_fea).to('cuda')
    model_eta = MLP(dim_fea).to('cuda')
    model_pyx, model_eta = pretrain(x_train, q_train, model_pyx, model_eta, lr, wd)

    accus = [ACCU(model_pyx(x_test).sigmoid().squeeze(), y_test).item()]
   
    optimizer = Adam([
        {'params': model_pyx.parameters()},
        {'params': model_eta.parameters()},
    ], lr = lr, weight_decay = wd)
    
    N = x_train.size(0)
    for e_step in range(20):
        
        P_yi1_xi = model_pyx(x_train).sigmoid().squeeze()
        eta      = model_eta(x_train).sigmoid().squeeze()
        pst = EStep(P_yi1_xi, eta, q_train).detach()

        for m_step in range(40):
            indies = t.randperm(N)[:int(N/10)]            
            optimizer.zero_grad()

            P_yi1_xi = model_pyx(x_train[indies]).sigmoid().squeeze()
            eta      = model_eta(x_train[indies]).sigmoid().squeeze()

            loss1 = -(pst[indies] * (t.stack([1-P_yi1_xi, P_yi1_xi], 1) + 1e-5).log()).sum(1)
            loss1 = loss1.topk(largest=False, k=int(loss1.size(0)*0.8))[0].mean()

            q1 = pst[:,1].squeeze()[indies]
            loss2 = -(q1 * ((((1-eta) ** (1-q_train[indies])) * (eta ** q_train[indies])).squeeze() + 1e-5).log()).mean()
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            accus.append(ACCU(model_pyx(x_test).sigmoid().squeeze(), y_test).item())

        
    #show(accus)
    return ACCU(model_pyx(x_test).sigmoid().squeeze(), y_test)

if __name__ == "__main__":
    
    # dataset reader
    # -------------------------------------------------------------------- #
    fea, tar, tar_pu, _ = syn(type=int(sys.argv[1]), noise_rate=float(sys.argv[2]), reader = reader_australian, Power=2)
    fea_norm = (fea - fea.mean(0, keepdim = True)) / fea.std(0, keepdim = True)
    print('p %.2f n %.2f' % (tar_pu[tar==1].mean(), (1-tar_pu)[tar==0].mean()))
    N = fea.size(0)
    print("feature dim %d" % fea.size(1))
    # -------------------------------------------------------------------- #

    # train
    # -------------------------------------------------------------------- #
    lr, wd = .001, 1e-2
    accu = []
    for i in range(5):
        shuffle = t.randperm(N)
        x, y, q = fea[shuffle], tar[shuffle], tar_pu[shuffle].float()
        x_train, q_train = x[:int(N *.8)], q[:int(N *.8)]
        x_test,  y_test  = x[int(N *.8):], y[int(N *.8):]
        temp = em_train(x_train, q_train, x_test,  y_test, lr, wd)
        accu.append(temp)
    # -------------------------------------------------------------------- #
    print('accu: %.4f std: %.4f | %.4f %.4f %.4f %.4f %.4f' % (t.Tensor(accu).mean().item(), t.Tensor(accu).std().item(), accu[0], accu[1], accu[2], accu[3], accu[4]))
    