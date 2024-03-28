import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn
def l1_reg(param):
    return torch.sum(torch.abs(param))

def softmax_entropy(param):
    # return - torch.sum(torch.special.entr(torch.abs(param)))  
    return - torch.sum(torch.special.entr(F.softmax(param,dim=0)))  

def mixuploss(x,y,criterion,net,alpha = 0.8):
    # mix_ratio = np.random.dirichlet(np.ones(3) * 0.9,size=1) # 设置为0.9
    lamda = np.random.beta(alpha,alpha)
    index = torch.randperm(x.size(0)).cuda()
    x = lamda*x + (1-lamda)*x[index,:]
    y = lamda*y + (1-lamda)*y[index,:]
    y_pre = net(x)
    if len(y_pre.shape) > 2:
        loss = criterion(y_pre,torch.max(y, 1)[1])
    else:
        loss = criterion(y_pre,y)
    return loss
    


class MultiLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=1):
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
        
    def forward(self, y_pred, y_true):
        # mse = torch.mean((y_true - y_pred) ** 2 / (y_true**2 + 1e-12))
                        
        # mae = self.mae_loss(y_pred, y_true)
        mape = torch.mean(torch.abs((y_true - y_pred) / y_true))
        # loss = self.alpha * mse + self.beta * mape 
        return mape
class Diffloss(nn.Module):
    def __init__(self,threshold = 0.01):
        super(Diffloss, self).__init__()
        self.threshold = threshold
    def forward(self, y_pred):
        y_t = y_pred[:,1:-1] # y_t
        y_t_1 = y_pred[:,0:-2] # y_t-1 > y_t
        diff =  y_t_1 - y_t
        monotonic_loss = torch.sum(torch.nn.functional.relu(diff-self.threshold)) + torch.sum(torch.nn.functional.relu(-diff))
           
        
        return monotonic_loss
    
class FMAELoss(nn.Module): # FOCAL mae
    def __init__(self, a = -1, T = 0.5):
        super().__init__()
        self.a = a
        self.T = T
        self.tau = 0.03
    def forward(self, inputs, targets):
        diff = torch.abs((inputs - targets) / targets)/self.T
        
        focal_term = (diff ** 2).sigmoid()**self.a
        early_term = diff.sigmoid()
        # exp_term = torch.exp(diff ** 2 / self.T)
        # denominator = 1 + exp_term
        # numerator = exp_term / denominator
        # loss = diff ** 2 * numerator ** self.a * torch.exp(diff / self.T) / (1 + torch.exp(diff / self.T))
        loss = (focal_term * early_term + self.tau) * diff ** 2
        return loss.mean()
    def set_a(self,a):
        self.a = a
    def set_T(self,T):
        self.T = T
        
class MMD_loss(nn.Module):
    
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target],dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
