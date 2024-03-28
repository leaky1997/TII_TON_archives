"""
Created on Mon OCT 10 19:49:25 2022
@author: Qi Li(李奇)
@Email: liq22@mails.tsinghua.edu.cn
"""

# common modules

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import sys
import copy
sys.path.append("../")
sys.path.append("./code")

# specific modules
from model.symbolic_base import symbolic_base,convlutional_operator,frequency_operation
from model.attention import attentionBlock
from sympy import Matrix, Function,simplify
from sympy import hadamard_product, HadamardProduct, MatrixSymbol

# from utils
from utils.file_utils import create_dir
from utils.model_utils import get_arity
from utils.symbolic_utils import prune_matrix,element_mul



def conv1x1(in_channels,out_channels,stride= 1,bias=True,groups=1):
    return nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias,groups=groups)
    
class neural_symbolc_base(nn.Module):
    '''
    neural symbolc net 基础
        symbolic_base = None, # 构成网络的符号空间,torch function的 list
        initialization = None,  # 初始化方法，如不给定则采用默认的方法，todo
        input_channel = 1, # 输入维度
        output_channel = None, # 输出维度，一般没用，因为输出维度是根据符号空间确定的
        bias = False, # shift
        device = 'cuda', # 是否使用GPU
        scale = 1, # 符号空间的倍数，例如 输入5维的符号空间，scale = 2 则符号空间为10
        
        该class 可以针对的input 为 [B,C,L], 如果求完均值后的特征可以是[B,C,1]
    '''
    def __init__(self,
                 symbolic_base = None,
                 input_channel = 1,
                 bias = False,
                 device = 'cuda',
                 scale = 1,
                 skip_connect = True,
                 amount = 0.5,
                 temperature = 1,
                 lenth = 2560,
                 args = None
                 ) -> None:
        super().__init__()
        
        self.symbolic_base_dic = symbolic_base
        self.input_channel = input_channel
        self.bias = bias
        self.device = device
        self.scale = scale
        self.skip_connect = skip_connect
        self.lenth = lenth
        self.device = device
        self.amount =  amount
        
        self.args = args
        
        self.set_t(temperature)
        self.__init_layers__()
    

    def __init_layers__(self):
        self.func = [self.symbolic_base_dic['torch'][f] for f in self.symbolic_base_dic['torch']] # 取出dic中的函数
        self.symbolic_base = self.func * self.scale
        
        self.function_arity = []
        self.layer_size = 0
        self.learnable_param = nn.ModuleList([])
        
        for k in range(self.scale):
            for i, (name,f) in enumerate(self.symbolic_base_dic['torch'].items()):
                
                index = i + k*len(self.symbolic_base_dic['torch'])
                
                if isinstance(f, convlutional_operator):  
                    conv_op = f
                    self.learnable_param.append(conv_op)
                    self.symbolic_base[index] = self.learnable_param[-1]
                if 'global' in name:
                    self.filter_op = f(length = self.lenth, type = name,device = self.device)
                    self.learnable_param.append(self.filter_op)
                    self.symbolic_base[index] = self.learnable_param[-1]  
                    self.register_parameter(f'learnable_param_{index}_weight', self.learnable_param[-1].weight)        ## 因为要拼接所以需要注册              
                elif  'fre_' in name:
                    self.filter_op = f(mu = 0.1, sigma = 0.1, length = self.lenth, type = name,device = self.device)
                    self.learnable_param.append(self.filter_op)
                    self.symbolic_base[index] = self.learnable_param[-1]
                    self.register_parameter(f'learnable_param_{index}_mu', self.learnable_param[-1].mu)   
                    self.register_parameter(f'learnable_param_{index}_sigma', self.learnable_param[-1].sigma)
                elif '_wave' in name:
                    self.filter_op = f(in_channels=1, signal_length=self.args.lenth, args=self.args) 
                    self.learnable_param.append(self.filter_op)
                    self.symbolic_base[index] = self.learnable_param[-1]
                    self.register_parameter(f'learnable_param_{index}f_c', self.learnable_param[-1].f_c)   
                    self.register_parameter(f'learnable_param_{index}f_b', self.learnable_param[-1].f_b)                      
                  
                                         
                arity = get_arity(self.symbolic_base[index])
                self.function_arity.append((self.symbolic_base[index],arity))
                self.layer_size += arity
                
        self.output_channel = int(len(self.symbolic_base))  
       
        self.In = nn.InstanceNorm1d(self.input_channel)
        self.channel_conv = nn.Conv1d(self.input_channel,self.layer_size,kernel_size=1,stride=1,padding=0,bias=self.bias)
        if self.skip_connect:
            self.down_conv = nn.Conv1d(self.input_channel,self.output_channel,kernel_size=1,stride= 1,padding=0,bias=self.bias)



    def weight_operator(self):
        pass
    
    def set_t(self,t):
        self.temperature = t
    def get_Matrix(self,x):
        '''
        x should be symbol
        '''

        self.sympy_func = [self.symbolic_base_dic['sympy'][f] for f in self.symbolic_base_dic['sympy']]
        self.sympy_symbolic_base = self.sympy_func * self.scale
        self.sympy_function_arity = []
        for f in self.sympy_symbolic_base:
            arity = get_arity(f)
            self.sympy_function_arity.append((f,arity))
      
        
        # 
        IN_layer = Matrix(x) # IN_layer = Matrix([Function('IN')(x_) for x_ in x])
        
        # argmax weight
        layer_weight = prune_matrix(self.channel_conv,amount = self.amount) 

        
        # channel
        
        args = layer_weight * IN_layer
        
        args_idx, signal = 0, []
        for i, (f, arity) in enumerate(self.sympy_function_arity):
            arg = args[args_idx: args_idx + arity]
            signal.append(f(*arg))
            args_idx = args_idx + arity
        signal = Matrix(signal)
        if self.skip_connect:
            # down_conv matrix
            layer_weight = prune_matrix(self.down_conv,amount = self.amount)
            signal_skip = layer_weight * IN_layer

            signal += signal_skip
        pooling = [s for s in signal] # pooling = [Function('pool')(s) for s in signal] # * 代表解开矩阵
        return pooling
      
        
        

        
    def forward(self,x): # B,C,L
        

        
        normed_x = self.In(x)
        # normed_x = avg_x
        self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=0) # 权重蒸馏 , dim=0
        # self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=1) # 权重蒸馏 , dim=1
        multi_channel_x = self.channel_conv(normed_x) # 相当于针对通道的全连接层
        
        args_idx = 0
        for i, (f, arity) in enumerate(self.function_arity):
            channel_x = multi_channel_x[:,args_idx: (args_idx + arity)]
            img = f(*torch.split(channel_x, 1, dim=1)) # split input + opearator + squeeze
            output = torch.cat((output, img), dim = 1) if i else  img
            args_idx += arity
            
            output = self.In(output) # b 不然梯度太小
            
        if self.skip_connect:

            output += self.down_conv(normed_x) #

        return output

class neural_feature_regression(neural_symbolc_base):
    def __init__(self, symbolic_base=None, input_channel=1,lenth = 1024, bias=False, device='cuda',
                 scale=1, skip_connect=True,amount = 0,temperature = 0.01,args = None) -> None: 
        super().__init__(symbolic_base, input_channel,
                         bias, device, scale,skip_connect,amount = amount,temperature = temperature,args = None)
    def forward(self,x):
        multi_channel_x = self.channel_conv(x)
        
        args_idx = 0
        for i, (f, arity) in enumerate(self.function_arity):
            channel_x = multi_channel_x[:,args_idx: (args_idx + arity)]
            img = f(*torch.split(channel_x, 1, dim=1)) # split input + opearator + squeeze
            output = torch.cat((output, img), dim = 1) if i else  img
            args_idx += arity

        return output        
    def get_Matrix(self,x):
        '''
        x should be symbol
        '''
        self.sympy_func = [self.symbolic_base_dic['sympy'][f] for f in self.symbolic_base_dic['sympy']]
        self.sympy_symbolic_base = self.sympy_func * self.scale
        self.sympy_function_arity = []
        for f in self.sympy_symbolic_base:
            arity = get_arity(f)
            self.sympy_function_arity.append((f,arity))

        # argmax weight
        layer_weight = prune_matrix(self.channel_conv,amount = self.amount)
        
        # channel
        
        args = layer_weight * Matrix(x)
        
        args_idx, signal = 0, []
        for i, (f, arity) in enumerate(self.sympy_function_arity):
            arg = args[args_idx: args_idx + arity]
            signal.append(f(*arg))
            args_idx = args_idx + arity
        signal = Matrix(signal)
        if self.skip_connect:
            # down_conv matrix
            layer_weight = prune_matrix(self.down_conv,amount = self.amount)
            signal_skip = layer_weight * Matrix(x)
            signal = signal + signal_skip
        pooling = [s for s in signal] # * 代表解开矩阵
        return pooling                

class neural_symbolic_regression(neural_symbolc_base):
    def __init__(self, symbolic_base=None, input_channel=1,lenth = 1024, bias=False, device='cuda',
                 scale=1, skip_connect=True,amount = 0,temperature = 0.01,args = None) -> None:
        super().__init__(symbolic_base, input_channel,
                         bias, device, scale,skip_connect,amount = amount,temperature = temperature,args = None)
    def forward(self,x):
        self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=0) # 权重蒸馏 , dim=0
        multi_channel_x = self.channel_conv(x)
        
        args_idx = 0
        for i, (f, arity) in enumerate(self.function_arity):
            channel_x = multi_channel_x[:,args_idx: (args_idx + arity)]
            img = f(*torch.split(channel_x, 1, dim=1)) # split input + opearator + squeeze
            output = torch.cat((output, img), dim = 1) if i else  img
            args_idx += arity
        if self.skip_connect:

            output += self.down_conv(x)

        return output        
    def get_Matrix(self,x):
        '''
        x should be symbol
        '''

        self.sympy_func = [self.symbolic_base_dic['sympy'][f] for f in self.symbolic_base_dic['sympy']]
        self.sympy_symbolic_base = self.sympy_func * self.scale
        self.sympy_function_arity = []
        for f in self.sympy_symbolic_base:
            arity = get_arity(f)
            self.sympy_function_arity.append((f,arity))
      

        layer_weight = prune_matrix(self.channel_conv,amount = self.amount)
        
        # channel
        
        args = layer_weight * Matrix(x)
        
        args_idx, signal = 0, []
        for i, (f, arity) in enumerate(self.sympy_function_arity):
            arg = args[args_idx: args_idx + arity]
            signal.append(f(*arg))
            args_idx = args_idx + arity
        signal = Matrix(signal)
        if self.skip_connect:
            # down_conv matrix
            layer_weight = prune_matrix(self.down_conv,amount = self.amount)
            signal_skip = layer_weight * Matrix(x)
            signal = signal + signal_skip
        pooling = [s for s in signal] # * 代表解开矩阵
        return pooling           

# %%          
class basic_model(nn.Module):
    def __init__(self,
                 input_channel = 1,
                 bias = False,
                 symbolic_bases = None,
                 scale = [1],
                 skip_connect = True,
                 down_sampling_kernel = None,
                 down_sampling_stride = 2,
                 lenth = 1,
                 num_class = 10,
                 device = 'cuda',
                 amount = 0.5,
                 temperature = 1
                 ) -> None:
        super().__init__()
        '''
        scale 和 symbolic_basede 尺度要对应
        '''     
        self.input_channel = input_channel
        self.bias = bias
        self.skip_connect = skip_connect
        self.amount = amount
        self.temperature = temperature

        self.scale = scale
        self.symbolic_bases = symbolic_bases

        self.down_sampling_kernel = down_sampling_kernel

        self.down_sampling_stride = down_sampling_stride
        
        self.lenth = lenth
        self.device = device
        
        self.to(self.device)
    def __make_layer__(self,
                       symbolic_bases,
                       input_channel = 1,
                       layer_type = 'transform' # 'regression'
                       ):            
        layers = []
        layer_selection = neural_symbolc_base if layer_type == 'transform' else neural_symbolic_regression
        
        for i,symbolic_base in enumerate(symbolic_bases):
            

                
                            
            next_channel = layer.output_channel if i else input_channel
            
            layer = layer_selection( symbolic_base = symbolic_base,
                 initialization = None,
                 input_channel = next_channel, 
                 output_channel = None,
                 bias = self.bias,
                 device = self.device,
                 scale = self.scale[i],
                 skip_connect = self.skip_connect,
                 lenth= self.lenth,
                 amount = self.amount,
                 temperature = self.temperature)            
            layers.append(layer)

                
        return nn.ModuleList(layers)
    
    def norm(self,x):
        mean = x.mean(dim = 0,keepdim = True)
        std = x.std(dim = 0,keepdim = True)
        out = (x-mean)/(std + 1e-10)
        return out
    
        
    def forward(self,x):

        for layer in self.symbolic_transform_layer:
            x = layer(x)  # 直接nan
        for layer in self.symbolic_feature_layer:
            self.feature = layer(x)

        x = self.feature
        for layer in self.symbolic_regression_layer:
            x = self.norm(x)
            x = layer(x)
            
        x = x.squeeze()
        x = self.regression_layer(x)

        return x
    
class DFN(basic_model):
    ## TODO to args
    def __init__(self, input_channel=1, bias=False, symbolic_bases=None, scale=[1],
     skip_connect=True, down_sampling_kernel=None, down_sampling_stride=2,lenth = 1024, num_class=10, device='cuda', amount=0.5,temperature = 1,
     expert_list = None, logic_list =None, feature_list = None, args = None) -> None:
        super().__init__(input_channel, bias, symbolic_bases, scale, skip_connect, down_sampling_kernel, down_sampling_stride,
                         lenth,num_class, device, amount,temperature)
        # 符号回归层
        self.expert_list = expert_list
        self.logic_list = logic_list
        self.feature_list = feature_list
        self.args = args
        
        self.symbolic_transform_layer = self.__make_layer__(symbolic_bases = expert_list,
                       input_channel= self.input_channel,
                       layer_type = 'transform')
        
        # 如果有降采样 则倒数第二层
        final_dim = self.symbolic_transform_layer[-1].output_channel
        
        
        self.symbolic_feature_layer = self.__make_layer__(symbolic_bases = [feature_list[0]],
                       input_channel= final_dim,
                       layer_type = 'feature')        
        
        feature_dim = self.symbolic_feature_layer[-1].output_channel
        
        # 符号回归层
        
        self.symbolic_regression_layer = self.__make_layer__(symbolic_bases = logic_list, # 1层符号回归层
                       input_channel= feature_dim,
                       layer_type = 'regression')
        
        final_dim = self.symbolic_regression_layer[-1].output_channel
        
        # 线性组合
        
        self.regression_layer = nn.Linear(final_dim,num_class,bias = bias)
        
        
        
        self.to(self.device)
    def __make_layer__(self,
                       symbolic_bases,
                       input_channel = 1,
                       layer_type = 'transform' # 'regression'
                       ):            
        layers = []
        if layer_type == 'transform':
            
            layer_selection = neural_symbolc_base 
            
        elif layer_type == 'regression' :
            layer_selection = neural_symbolic_regression
            
        elif layer_type == 'feature':
            layer_selection = neural_feature_regression
        
        for i,symbolic_base in enumerate(symbolic_bases):
                
                            
            next_channel = layer.output_channel if i else input_channel
            
            layer = layer_selection( symbolic_base = symbolic_base,
                 input_channel = next_channel, 
                 bias = self.bias,
                 device = self.device,
                 scale = self.scale,
                 skip_connect = self.skip_connect,
                 amount = self.amount,
                 lenth = self.lenth,
                 temperature = self.temperature,
                 args = self.args)            
            layers.append(layer)

                
        return nn.ModuleList(layers)
    def forward(self,x):

        for layer in self.symbolic_transform_layer:
            x = layer(x)  # 直接nan
        for layer in self.symbolic_feature_layer:
            self.feature = layer(x)

        x = self.feature
        for layer in self.symbolic_regression_layer:
            x = self.norm(x)
            x = layer(x)
            
        x = x.view(x.size(0), -1)
        x = self.regression_layer(x)

        return x
    


#%%    
if __name__ == '__main__':  
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    save_dir = './save_dir'
    create_dir(save_dir)
    summaryWriter = SummaryWriter(save_dir + "/test")
    x = torch.randn(32,1,1024, dtype= torch.float32).cuda()
    
    base = symbolic_base(['add','mul','sin','exp','idt','sig','tanh','pi','e']) # 符号空间
    

#%% test DNSN
    expert_list = symbolic_base(['envelope','fre_normal','fre_sinc'])
    
    feature_list = symbolic_base(['entropy','kurtosis','mean'])
    
    logic_list= symbolic_base(['imp'])
    
    model = DFN(input_channel = 1,
                 bias = False,
                 symbolic_bases = [base,base,base], 
                 scale = 4,
                 skip_connect = True,
                 down_sampling_kernel = 1,
                 down_sampling_stride = 2,
                 lenth = 1024,
                 num_class = 3,
                 expert_list = [expert_list,expert_list,expert_list],
                 feature_list = [feature_list,feature_list,feature_list],
                 logic_list = [logic_list,logic_list]
                 )
    y = model(x)
 
            
    print(summary(model,(1,1024),device = "cuda"))

        