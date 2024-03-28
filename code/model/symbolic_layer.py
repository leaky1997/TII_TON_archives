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
    
    # def deepcopy(self,func,scale):
    #     cache = []
    #     for i in range(scale):
    #         cache += copy.deepcopy(func)
    #     return cache

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

            
            # for i,f in enumerate(self.symbolic_base):
            #     if isinstance(f, convlutional_operator):
            #         conv_op = f
            #         self.learnable_param.append(conv_op)
            #         self.symbolic_base[i] = (lambda x : conv_op(x))  
            #     if f == frequency_operation:
            #         filter_op = f(mu = 0.1, sigma = 0.1, length = self.lenth, type = 'normal',device = self.device)
            #         self.learnable_param.append(filter_op)
            #         self.symbolic_base[i] = (lambda x : filter_op(x)) 
                  
                                         
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
        
        # IN_layer = IN_layer*len(x)
        
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
            # signal = hadamard_product(signal,signal_skip) + signal_skip
            signal += signal_skip
        pooling = [s for s in signal] # pooling = [Function('pool')(s) for s in signal] # * 代表解开矩阵
        return pooling
      
        
        

        
    def forward(self,x): # B,C,L
        
        # avg_x = self.moving_avg_layer(x)
        
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
            # self.down_conv.weight.data = F.softmax((1.0 / self.temperature) * self.down_conv.weight.data, dim=0) # 权重蒸馏 , dim=0
            # self.down_conv.weight.data = F.softmax((1.0 / self.temperature) * self.down_conv.weight.data, dim=1) # 权重蒸馏 , dim=1
            output += self.down_conv(normed_x) #
            # output = output * self.down_conv(normed_x) + self.down_conv(normed_x) #
            # output += normed_x
        # output = self.decomposition_sin(output)
        # output = self.decomposition_sin(output)
        # output = self.moving_avg_layer(output)
        # output = self.CA(output)  #
        return output

class neural_feature_regression(neural_symbolc_base):
    def __init__(self, symbolic_base=None, input_channel=1,lenth = 1024, bias=False, device='cuda',
                 scale=1, skip_connect=True,amount = 0,temperature = 0.01,args = None) -> None: 
        super().__init__(symbolic_base, input_channel,
                         bias, device, scale,skip_connect,amount = amount,temperature = temperature,args = None)
    def forward(self,x):
        # self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=0) # 权重蒸馏 , dim=0
        # self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=1) # 权重蒸馏 , dim=1        
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
      
        
        # 
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
        # self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=1) # 权重蒸馏 , dim=1        
        multi_channel_x = self.channel_conv(x)
        
        args_idx = 0
        for i, (f, arity) in enumerate(self.function_arity):
            channel_x = multi_channel_x[:,args_idx: (args_idx + arity)]
            img = f(*torch.split(channel_x, 1, dim=1)) # split input + opearator + squeeze
            output = torch.cat((output, img), dim = 1) if i else  img
            args_idx += arity
        if self.skip_connect:
            # self.down_conv.weight.data = F.softmax((1.0 / self.temperature) * self.down_conv.weight.data, dim=0) # 权重蒸馏 , dim=0
            # self.down_conv.weight.data = F.softmax((1.0 / self.temperature) * self.down_conv.weight.data, dim=1) # 权重蒸馏 , dim=1            
            output += self.down_conv(x)
            # output += x
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
      
        
        # 
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
#%%  todo     

# %%          
class basic_model(nn.Module):
    def __init__(self,
                 input_channel = 1,
                 bias = False,
                 symbolic_bases = None,
                #  symbolic_bases_4reg = None,
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
        # assert len(symbolic_bases) == len(scale)
        self.scale = scale
        self.symbolic_bases = symbolic_bases
        # self.symbolic_bases_4reg = symbolic_bases_4reg
        self.down_sampling_kernel = down_sampling_kernel

        self.down_sampling_stride = down_sampling_stride
        
        self.lenth = lenth
        self.device = device
        # 符号变换层
        # self.moving_avg_layer = nn.AvgPool1d(kernel_size = 8,
        #                                            stride = 8,
        #                                            padding=(8-1)//2)  # ztodo
        
        # self.symbolic_transform_layer = self.__make_layer__(symbolic_bases = self.symbolic_bases,
        #                input_channel= self.input_channel,
        #                layer_type = 'transform')
        
        # # 如果有降采样 则倒数第二层
        # final_dim = self.symbolic_transform_layer[-1].output_channel
        
        # # 符号回归层
        
        # self.symbolic_regression_layer = self.__make_layer__(symbolic_bases = [self.symbolic_bases[0]], # 1层符号回归层
        #                input_channel= final_dim,
        #                layer_type = 'regression')
        
        # final_dim = self.symbolic_regression_layer[-1].output_channel
        
        # # 线性组合
        
        # self.regression_layer = nn.Linear(final_dim,num_class,bias = bias)
        
        self.to(self.device)
    def __make_layer__(self,
                       symbolic_bases,
                       input_channel = 1,
                       layer_type = 'transform' # 'regression'
                       ):            
        layers = []
        layer_selection = neural_symbolc_base if layer_type == 'transform' else neural_symbolic_regression
        
        for i,symbolic_base in enumerate(symbolic_bases):
            
            # if down_sampling_kernel is not None :
                
            #     layers.append(moving_avg_layer)
                
                            
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
    
    # def softmax(self,x):
    #     return nn.Softmax(dim=1)(x)
        
    def forward(self,x):

        for layer in self.symbolic_transform_layer:
            x = layer(x)  # 直接nan
        for layer in self.symbolic_feature_layer:
            self.feature = layer(x)
        # # x = signal*x + x
        # self.feature = x.mean(dim=-1, keepdim = True)
        x = self.feature
        for layer in self.symbolic_regression_layer:
            x = self.norm(x)
            x = layer(x)
            
        x = x.squeeze()
        x = self.regression_layer(x)
        # x = nn.Softmax(dim=1)(x)
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
        
        # self.moving_avg_layer = nn.AvgPool1d(kernel_size = down_sampling_kernel,
        #                                            stride = down_sampling_stride,
        #                                            padding=(down_sampling_kernel)//2)
        
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
            
            # if down_sampling_kernel is not None :
                
            #     layers.append(moving_avg_layer)
                
                            
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
        # # x = signal*x + x
        # self.feature = x.mean(dim=-1, keepdim = True)
        x = self.feature
        for layer in self.symbolic_regression_layer:
            x = self.norm(x)
            x = layer(x)
            
        x = x.view(x.size(0), -1)
        x = self.regression_layer(x)
        # x = nn.Softmax(dim=1)(x)
        return x
    
# from functools import partial
# from einops.layers.torch import Rearrange, Reduce
# from einops import rearrange
# import math
# from torch import nn, einsum

# class T5RelativePositionBias(nn.Module): 
#     def __init__(
#         self,
#         scale,
#         causal = False,
#         num_buckets = 32,
#         max_distance = 128
#     ):
#         super().__init__()
#         self.scale = scale
#         self.causal = causal
#         self.num_buckets = num_buckets
#         self.max_distance = max_distance
#         self.relative_attention_bias = nn.Embedding(num_buckets, 1)

#     @staticmethod
#     def _relative_position_bucket(
#         relative_position,
#         causal = True,
#         num_buckets = 32,
#         max_distance = 128
#     ):
#         ret = 0
#         n = -relative_position
#         if not causal:
#             num_buckets //= 2
#             ret += (n < 0).long() * num_buckets
#             n = torch.abs(n)
#         else:
#             n = torch.max(n, torch.zeros_like(n))

#         max_exact = num_buckets // 2
#         is_small = n < max_exact

#         val_if_large = max_exact + (
#             torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
#         ).long()
#         val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

#         ret += torch.where(is_small, n, val_if_large)
#         return ret

#     def forward(self, x):
#         i, j, device = *x.shape[-2:], x.device
#         q_pos = torch.arange(i, dtype = torch.long, device = device)
#         k_pos = torch.arange(j, dtype = torch.long, device = device)
#         rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
#         rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
#         values = self.relative_attention_bias(rp_bucket)
#         bias = rearrange(values, 'i j 1 -> i j')
#         return bias * self.scale
# class OffsetScale(nn.Module):
#     def __init__(self, dim, heads = 1):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(heads, dim))
#         self.beta = nn.Parameter(torch.zeros(heads, dim))
#         nn.init.normal_(self.gamma, std = 0.02)

#     def forward(self, x):
#         out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
#         return out.unbind(dim = -2)    
# class SingleHeadedAttention(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         dim_qk,
#         dim_value,
#         causal = False,
#     ):
#         super().__init__()
#         self.causal = causal


#         self.attn_fn = partial(F.softmax, dim = -1) 

#         self.rel_pos_bias = T5RelativePositionBias(causal = causal, scale = dim_qk ** 0.5)

#         self.to_qk = nn.Sequential(
#             nn.Linear(dim, dim_qk),
#             nn.SiLU()
#         )

#         self.offsetscale = OffsetScale(dim_qk, heads = 2)

#         self.to_v = nn.Sequential(
#             nn.Linear(dim, dim_value),
#             nn.SiLU()
#         )
#         self.weight_x = nn.Parameter(torch.randn(512, dim, 1, dtype=torch.float32) * 0.02)
#         self.norm = nn.LayerNorm(dim)
        
#     def forward(self, x):
#         seq_len, dim, device, dtype = *x.shape[-2:], x.device, x.dtype # (..., seq_len, dim)

#         v_input = x # if v_input is None, use x
        
#         # a = b = int(math.sqrt(seq_len)) # assume that seq_len is a perfect square
        
#         # v_input = v_input.view(-1, a, b, dim)
        
#         v_input = torch.fft.rfft(x, dim=2, norm='ortho') # 
#         # weight = torch.view_as_complex(self.complex_weight_x)
#         v_input = v_input * self.weight_x
#         v_input = torch.fft.ifft2(v_input, dim=2, norm='ortho')
#         v_input = v_input.view(-1, seq_len, dim).real
        
        

#         qk, v = self.to_qk(x), self.to_v(v_input) # (..., seq_len, dim_qk), (..., seq_len, dim_value)
#         q, k = self.offsetscale(qk) # 

#         scale =  dim ** -0.5

#         sim = einsum('b i d, b j d -> b i j', q, k) * scale

#         sim = sim + self.rel_pos_bias(sim)

#         if self.causal:
#             causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).triu(1)

#         if self.causal and not self.laplacian_attn_fn:
#             # is softmax attention and using large negative value pre-softmax
#             sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

#         attn = self.attn_fn(sim)

#         if self.causal and self.laplacian_attn_fn:
#             # if using laplacian attention function, zero out upper triangular with 0s
#             attn = attn.masked_fill(causal_mask, 0.)

#         return einsum('b i j, b j d -> b i d', attn, v) # (..., seq_len, dim_value) 
    
       #### 不做RUL
# class DFN_rul(DFN):
#     def __init__(self, input_channel=1, bias=False, symbolic_bases=None, scale=[1],
#      skip_connect=True, down_sampling_kernel=None, down_sampling_stride=2,lenth = 1024, num_class=10, device='cuda', amount=0.5,temperature = 1,
#      expert_list = None, logic_list =None, feature_list = None) -> None:
#         super().__init__(input_channel, bias, symbolic_bases, scale, skip_connect, down_sampling_kernel, down_sampling_stride,
#                          lenth,num_class, device, amount,temperature, expert_list, logic_list, feature_list)
#         # self.regression_layer = nn.Linear(self.regression_layer.output_channel,1,bias = bias)
        
        
#         self.nonlinear = nn.Sequential(
#             nn.Linear(self.symbolic_regression_layer[-1].output_channel, 1),
#             nn.SiLU(),
#             # nn.Dropout(0.2),
#             # nn.Linear(128, 64),
#             # nn.SiLU(),
#             # nn.Dropout(0.2),            
#             # nn.Linear(64, 1),
#             # nn.Sigmoid()
#         )
#         self.to(self.device)
        
#     def forward(self,x):

#         for layer in self.symbolic_transform_layer:
#             x = layer(x)  # 直接nan
#         for layer in self.symbolic_feature_layer:
#             x = torch.fft.fft(x, dim=2, norm='ortho').real
#             x = layer(x)
#         # # x = signal*x + x
#         # self.feature = x.mean(dim=-1, keepdim = True)

#         for layer in self.symbolic_regression_layer:
#             x = self.norm(x) #+ t  样本norm
#             x = layer(x)
            
#         self.feature = x.squeeze()
        

#         x = self.nonlinear(self.feature)



#         # x = nn.Softmax(dim=1)(x)
#         return x


#%%    
if __name__ == '__main__':  
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    save_dir = './save_dir'
    create_dir(save_dir)
    summaryWriter = SummaryWriter(save_dir + "/test")
    x = torch.randn(32,1,1024, dtype= torch.float32).cuda()
    
    base = symbolic_base(['add','mul','sin','exp','idt','sig','tanh','pi','e']) # ['add','mul','sin','exp','idt','sig','tanh','pi','e']
    
#%% 测试base 模块
    # model = neural_symbolc_base(
    #     symbolic_base = base,
    #     initialization = None,
    #     input_channel = 1,
    #     output_channel = None,
    #     bias = False,
    #     device = 'cuda',
    #     scale = 1 
    # )
    # y = model(x)
    # print(summary(model,(1,1024),device = "cuda"))
    
#%% 测试完整的layer

    # model = basic_model(input_channel = 1,
    #              bias = False,
    #              symbolic_bases = [base,base,base],
    #             #  symbolic_bases_4reg = None,
    #              scale = [4,4,4],
    #              skip_connect = True,
    #              down_sampling_kernel = [8,8,8],
    #              down_sampling_stride = [2,2,2],
    #              num_class = 10,)
    # y = model(x)
 
            
    # print(summary(model,(1,1024),device = "cuda"))
#%% test DNSN
    expert_list = symbolic_base(['envelope','fre_normal','fre_sinc'])
    
    feature_list = symbolic_base(['entropy','kurtosis','mean'])
    
    logic_list= symbolic_base(['imp'])
    
    model = DFN(input_channel = 1,
                 bias = False,
                 symbolic_bases = [base,base,base],
                #  symbolic_bases_4reg = None,
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
#%% test global prune
    # layers = []
    # for i, module in enumerate(model.children()):
    #     if not isinstance(module, nn.Sequential):
    #         layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module] 

    
    # parameters_to_prune = ()
    # for layer in layers:
    #     if isinstance(layer, neural_symbolc_base):
            
    #         parameters_to_prune = *parameters_to_prune,(layer.channel_conv,"weight") 
    #         parameters_to_prune = *parameters_to_prune,(layer.down_conv,"weight")
            
       
    # import torch.nn.utils.prune as prune  
    
    # prune.global_unstructured(
    # parameters_to_prune,
    # pruning_method=prune.L1Unstructured,
    # amount=0.2,
    # )
        