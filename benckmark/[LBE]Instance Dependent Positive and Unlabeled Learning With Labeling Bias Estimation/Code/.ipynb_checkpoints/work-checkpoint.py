import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
data = pd.read_csv('./code/data/banknote.txt', names=column_names)

x = data.drop('class', axis=1)
y = data['class']

logistic_model = LogisticRegression()

logistic_model.fit(x, y)

y_pred = logistic_model.predict(x)

accuracy = accuracy_score(y, y_pred)
print(f'Accuracy of the logistic regression model: {accuracy:.2f}')

print('Model coefficients (θ):', logistic_model.coef_) # Theta
print('Model intercept (θ0):', logistic_model.intercept_)

theta_lgt = logistic_model.coef_[0]


def strategy1(x,theta_lgt,k):
    eta_x = (1 + np.exp(-np.dot(theta_lgt.T, x)))**(-1/k)
    return eta_x

def strategy2(x,theta_lgt,k):
    eta_x = 1 - strategy1(x,theta_lgt,k)
    return eta_x


x_values = x.values

k = 10
pi = 0.2
data['eta'] = [strategy2(x_i, theta_lgt, k) for x_i in x_values]

data_with_pu_label = data.sort_values(by='eta', ascending=False)

nagetive = sum(data_with_pu_label['class']==0)

S_PU = int(nagetive*pi/(1-pi)) # S_PU是根据pi计算S_PU集合中，正例的数量来进行数据划分


# 选择前 len(data_with_pu_label) - S_PU 的样本作为正例
p_threshold = len(data_with_pu_label) - S_PU - nagetive
data_with_pu_label['pu_label'] = 0  # 创建一个新列以存储 PU 标签
data_with_pu_label.iloc[:p_threshold, -1] = 1  # 前设置为观测到的正例


data_group = data_with_pu_label.groupby('pu_label')
S_P = data_group.get_group(1)
S_U = data_group.get_group(0)

S_P = S_P.iloc[:, :4]
S_P['y1'] = 1
S_P['y0'] = 0
S_P['si'] = 1
S_P['yi'] = 1

S_U = S_U.iloc[:, :4]
S_U['y1'] = 0 # 这里S_U在进入EM算法前应该根据已初始化的theta值，计算出widetilde{P(y_i)}的值
S_U['y0'] = 0
S_U['si'] = 0
S_U['yi'] = 3 # 用3 表示 Unknown





xi_SU = torch.tensor(S_U.drop(['y1','y0','si','yi'], axis=1).values, dtype=torch.float32)
y1_SU = torch.tensor(S_U['y1'].values, dtype=torch.int64)
y0_SU = torch.tensor(S_U['y0'].values, dtype=torch.int64)
si_SU = torch.tensor(S_U['si'].values, dtype=torch.int64)
yi_SU = torch.tensor(S_U['yi'].values, dtype=torch.int64)



S_U = [[xi_SU[i], y1_SU[i],y0_SU[i],si_SU[i],yi_SU[i]] for i in range(len(S_U))]

xi_SP = torch.tensor(S_P.drop(['y1','y0','si','yi'], axis=1).values, dtype=torch.float32)
y1_SP = torch.tensor(S_P['y1'].values, dtype=torch.int64)
y0_SP = torch.tensor(S_P['y0'].values, dtype=torch.int64)
si_SP = torch.tensor(S_P['si'].values, dtype=torch.int64)
yi_SP = torch.tensor(S_P['yi'].values, dtype=torch.int64)



S_P = [[xi_SP[i], y1_SP[i],y0_SP[i],si_SP[i],yi_SP[i]] for i in range(len(S_P))]

# init theta1 theta2

x = data_with_pu_label.iloc[:, :4]
y = data_with_pu_label['pu_label']

logistic_model = LogisticRegression()
logistic_model.fit(x, y)

theta1 = torch.tensor(logistic_model.coef_[0],dtype=torch.float32)
theta2 = torch.zeros(4,dtype=torch.float32)


import torch
import torch.optim as optim

# 定义EM算法模型类
class PU_EM_Model:
    def __init__(self, theta1, theta2,S_U,S_P):
        """
        dataset = [x_i(4D),y_1,y_0,s_i]
        """
        
        self.S_U = S_U
        self.S_P = S_P
        self.dataset = S_U+S_P
        self.theta1 = theta1.clone().detach().requires_grad_(True)
        self.theta2 = theta2.clone().detach().requires_grad_(True)

    def _logistic_function(self,theta, x):
        # logistic function
        return 1.0 / (1.0 + torch.exp(-torch.matmul(theta.T, x)))

    def eta(self,x):
        # 公式(8)
        return self._logistic_function(self.theta2, x)
        
    def h(self,x):
        # 公式(9)
        return self._logistic_function(self.theta1, x)

    def grad_expressions_theta1(self):
        grad_theta1 = torch.zeros_like(self.theta1)
        for data in self.dataset:
            x = data[0]
            y_1 = data[1]
            y_0 = data[2] 
            s = data[3]
            
            h_x = self.h(x)
            grad_theta1 += y_1 * (h_x - 1) * x + y_0 * h_x * x
        return grad_theta1
            
            
    def grad_expressions_theta2(self):
        grad_theta2 = torch.zeros_like(self.theta2)
        for data in self.dataset:
            x = data[0]
            y_1 = data[1]
            y_0 = data[2]
            s = data[3]
            yi =data[4]

            if yi == 1:  # 仅当 y_i == 1 时计算梯度贡献
                # grad_theta2 += ((-1) ** (s + 1)) * p_yi / p_si_given_yi_xi * eta_x * (eta_x - 1) * x
                grad_theta2 += (self.eta(x)-1)*x
        
                
        return grad_theta2

    
    def expectation_step(self):
        # 在E步中更新隐变量的期望
        for i in range(len(self.dataset)):
            # 为了可读性
            x = self.dataset[i][0]
            y_1 = self.dataset[i][1]
            y_0 = self.dataset[i][2]
            s = self.dataset[i][3]
            
            if not s.item():
                # s 为1 的情况下什么都不需要改变
                # s 为 0的情况下 ,通过公式(12)对\cap{P(y)}的值进行更新
                p_y1 = (1-self.eta(x))*self.h(x)
                p_y0 = 1-self.h(x)
                norm_factor = 1-self.h(x)*self.eta(x)
                # 更新数据集中的 y_1 和 y_0
                self.dataset[i][1] = p_y1
                self.dataset[i][2] = p_y0


        
        
    def maximization_step(self, optimizer):
        # 在M步中更新参数 theta1 和 theta2
        optimizer.zero_grad()
        # 计算梯度表达式
        self.theta1.grad = self.grad_expressions_theta1()  # 计算 theta1 的梯度
        self.theta2.grad = self.grad_expressions_theta2()  # 计算 theta2 的梯度

        optimizer.step()

model = PU_EM_Model(theta1, theta2,S_U,S_P)

# 定义优化器参数
lr = 0.01  # 学习率
rho1 = 0.9  # beta1，用于一阶矩估计的衰减率
rho2 = 0.999  # beta2，用于二阶矩估计的衰减率
optimizer = optim.Adam([model.theta1, model.theta2], lr=lr, betas=(rho1, rho2))

# 
def train_model(model,num_epochs):
    for epoch in range(num_epochs):
        # E步
        model.expectation_step()
        
        # M步
        model.maximization_step(optimizer)
        
        # 输出当前参数
        # print(f'Epoch [{epoch+1}/{num_epochs}], theta1: {model.theta1}, theta2: {model.theta2}')



# # 训练模型
train_model(model,num_epochs=100)