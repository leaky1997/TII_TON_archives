import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from math import pi
import torch.nn.functional as F
from .symbolic_layer import neural_symbolc_base,basic_model,neural_symbolic_regression
from .symbolic_base import feature_base, symbolic_base # 去掉了featurebase
#%% resnet Morlet
def Morlet(p):
    C = pow(pi, 0.25)
    # p = 0.03 * p
    y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)
    return y

class Morlet_fast(nn.Module):

    def __init__(self, num_classs, kernel_size, in_channels=1):

        super(Morlet_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.num_classs = num_classs
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, num_classs)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 10, num_classs)).view(-1, 1)
        
        # self.register_parameter('param_a', self.a_) # should
        # self.register_parameter('param_b', self.b_)

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right.cuda() - self.b_.cuda() / self.a_.cuda()
        p2 = time_disc_left.cuda() - self.b_.cuda() / self.a_.cuda()

        Morlet_right = Morlet(p1)
        Morlet_left = Morlet(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.num_classs, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)

# -----------------------input size>=32---------------------------------
def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, num_class=4, zero_init_residual=False, Wave_first = True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Morlet_fast(64, 16) if Wave_first else nn.Conv1d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




class ResNet_1_4(nn.Module):

    def __init__(self, block, layers, in_channel=1, num_class=4, zero_init_residual=False, Wave_first = True):
        super(ResNet_1_4, self).__init__()
        self.inplanes = 64
        self.conv1 = Morlet_fast(64, 16) if Wave_first else nn.Conv1d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        self.signal = x
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        self.feature = x
        x = self.fc(x)
        
        return x
    

# 
class f_eql(nn.Module):
    def __init__(self, input_channel=1, bias=False, symbolic_bases=None, scale=...,
     skip_connect=True, down_sampling_kernel=None, down_sampling_stride=2,
      num_class=4, device='cuda', amount=0.5) -> None:
        super(f_eql,self).__init__()
        
        self.input_channel = input_channel
        self.bias = bias
        self.skip_connect = skip_connect
        self.amount = amount
        
        # assert len(symbolic_bases) == len(scale)
        self.scale = scale
        self.symbolic_bases = feature_base()
        self.symbolic_bases_4reg = symbolic_base(['mul','sin','exp','idt','sig','tanh','pi','e'])
        self.down_sampling_kernel = down_sampling_kernel
        if self.down_sampling_kernel is not None :
            assert len(scale) == len(self.down_sampling_kernel), 'dimention mismatch'
        self.down_sampling_stride = down_sampling_stride
        self.device = device
        # 符号变换层
        
        self.symbolic_transform_layer = self.__make_layer__(symbolic_bases = [self.symbolic_bases],
                       input_channel= self.input_channel,
                       layer_type = 'transform')
        
        # 如果有降采样 则倒数第二层
        final_dim = self.symbolic_transform_layer[-1].output_channel
        
        # 符号回归层
        
        self.symbolic_regression_layer = self.__make_layer__(symbolic_bases = [self.symbolic_bases_4reg], # 1层符号回归层
                       input_channel= final_dim,
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
        layer_selection = neural_symbolc_base if layer_type == 'transform' else neural_symbolic_regression
        
        for i,symbolic_base in enumerate(symbolic_bases):
                
                            
            next_channel = layer.output_channel if i else input_channel
            
            layer = layer_selection( symbolic_base = symbolic_base,
                #  initialization = None,
                 input_channel = next_channel, 
                #  output_channel = None,
                 bias = self.bias,
                 device = self.device,
                 scale = 1,
                #  kernel_size = 1,
                #  stride = 1,
                 skip_connect = self.skip_connect,
                 amount = self.amount)            
            layers.append(layer)

                
        return nn.ModuleList(layers)
    
    def norm(self,x):
        mean = x.mean(dim = 1,keepdim = True)
        std = x.std(dim = 1,keepdim = True)
        out = (x-mean)/(std + 1e-10)
        return out        
    
    def forward(self,x):
        for layer in self.symbolic_transform_layer:
            x = layer(x)  # 直接nan
        self.feature = x.mean(dim=-1, keepdim = True)
        x = self.feature
        for layer in self.symbolic_regression_layer:
            x = self.norm(x)
            x = layer(x)
            
        x = x.squeeze()
        x = self.regression_layer(x)
        # x = nn.Softmax(dim=1)(x)
        return x
#%% ELM with WT + HT + FT + feature + ELM
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableWaveletLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LearnableWaveletLayer, self).__init__()
        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize learnable parameters
        self.w = nn.Parameter(torch.randn(out_channels, 1))
        self.fc = nn.Parameter(torch.randn(out_channels, 1))
        
        # self.register_parameter('w', self.w)
        # self.register_parameter('fc', self.fc)
        
    def forward(self, x):
        # Compute wavelet coefficients
        
        n = torch.arange(-self.kernel_size//2, self.kernel_size//2, dtype=torch.float32, device=x.device)
        psi = self.w / torch.sqrt(torch.tensor(torch.pi, device=x.device)) * torch.exp(-self.w**2 * n**2) * torch.exp(1j * 2 * torch.pi * self.fc * n) #
        psi = psi.view(self.out_channels, 1, self.kernel_size)
        psi = psi.repeat(1, self.in_channels, 1)
        # psi = psi.view(self.out_channels*self.in_channels, 1, self.kernel_size)
        

        x = nn.functional.conv1d(x, psi.real, groups=self.out_channels*self.in_channels)

        
        return x

def env_hilbert(x):
    '''
        Perform Hilbert transform along the last axis of x.
        
        Parameters:
        -------------
        x (Tensor) : The signal data. 
                     The Hilbert transform is performed along last dimension of `x`.
        
        Returns:
        -------------
        analytic (Tensor): A complex tensor with the same shape of `x`,
                           representing its analytic signal. 
                           
        Hilbert transform @ wiki: https://en.wikipedia.org/wiki/Hilbert_transform
            
    '''
    
    x = torch.as_tensor(x) # .double()
    
    N = x.shape[-1]
    Xf = torch.fft.fft(x)
    if (N % 2 == 0):
        Xf[..., 1 : N//2] *= 2
        Xf[..., N//2+1 : ] = 0
    else:
        Xf[..., 1 : (N+1)//2] *= 2
        Xf[..., (N+1)//2 : ] = 0
    return torch.fft.ifft(Xf).abs() # squ()


class ELM(nn.Module):
    def __init__(self, input_size, h_size, num_classes):
        super(ELM, self).__init__()
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = num_classes

        self._alpha = nn.Parameter(torch.empty(self._input_size, self._h_size).uniform_(-1., 1.))
        self._beta = nn.Parameter(torch.empty(self._h_size, self._output_size).uniform_(-1., 1.))
        self._bias = nn.Parameter(torch.zeros(self._h_size))

        self._activation = nn.Sigmoid()
        
        self.register_parameter('alpha', self._alpha)
        self.register_parameter('beta', self._beta)
        self.register_parameter('bias', self._bias)
        
        # self.bn = nn.BatchNorm1d(self._h_size)

    def forward(self, x):
        h = self._activation(torch.add(x @ self._alpha, self._bias))
        
        # h = self.bn(h)
        
        out = h @ self._beta

        return out.view(-1, self._output_size)

class Dong_ELM(nn.Module):
    def __init__(self, num_class = 4):
        super(Dong_ELM, self).__init__()
        
        # self.DWT0= DWT1DForward(J=1, wave='morlet').cuda()
        self.wave_t = LearnableWaveletLayer(1,1, 32)
        
        self.hilbert_t = env_hilbert
        
        self.mean = lambda x: torch.mean(x,dim=-1, keepdim = True)
        
        self.entropy = lambda x: (x * torch.log(x)).mean(dim=-1, keepdim = True)
       
        self.kurtosis = lambda x: (((x - torch.mean(x,dim=-1, keepdim = True)) ** 4).mean(dim=-1, keepdim = True)) / ((torch.var(x,dim=-1, keepdim = True) ** 2).mean(dim=-1, keepdim = True))
        
        self.gini = lambda x:  (torch.sort(x)[0] * (-2 * x.shape[-1] + 2 * torch.arange(x.shape[-1], device=x.device) - 1) / x.shape[-1]).mean(dim=-1, keepdim = True)
        # self.gini = lambda x:  torch.sum(torch.sort(x)[0] * (-2 * x.shape[-1] + 2 * torch.arange(x.shape[-1], device=x.device) - 1) / x.shape[-1], keepdim=True)
        
        def reciprocal_smoothness(x):
            # 除了NSE，还需要原始信号计算系数
            x = x**2
            NSES = x / torch.sum(x, dim=-1, keepdim=True)
            B = x.shape[0]
            a = (x / torch.pow(torch.prod(x, dim=-1, keepdim=True) + 1e-6, 1/B).mean(dim=-1, keepdim=True))
            res = torch.sum(NSES * a, dim=-1, keepdim=True)
            return res
        
        self.reciprocal_smoothness = reciprocal_smoothness
        

        
        self.elm = ELM(4, 1024, num_class)
        
    def forward(self, input):

        wave = self.wave_t(input)
        env = self.hilbert_t(wave)
        NSE = env**2 / torch.sum(env**2, dim=-1, keepdim=True)
        
        kurtosis = self.kurtosis(NSE)
        entropy = self.entropy(NSE)
        gini = self.gini(NSE)
        reciprocal_smoothness = self.reciprocal_smoothness(env)
        
        features = torch.cat([kurtosis, entropy, gini, reciprocal_smoothness], dim=-1)
        
        features = F.relu(features)
        
        output = self.elm(features)
        
        return output
    
# if __name__ == '__main__':
#     from torchsummary import summary
#     model = Dong_ELM()
#     summary(model.cuda(), (1, 64))
#     x = torch.randn(2,1,1024).cuda()
#     y = model(x)
#     print(y.shape)

## https://github.com/qthequartermasterman/torch_pso


#%% wanghuan‘ work


class A_cSE(nn.Module):
    
    def __init__(self, in_ch):
        super(A_cSE, self).__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, int(in_ch/2), kernel_size=1, padding=0),
            nn.BatchNorm1d(int(in_ch/2)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(int(in_ch/2), in_ch, kernel_size=1, padding=0),
            nn.BatchNorm1d(in_ch)
        )
        
    def forward(self, in_x):
        
        x = self.conv0(in_x)
        x = nn.AvgPool1d(x.size()[2:])(x)
        #print('channel',x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        
        return in_x * x + in_x

class SConv_1D(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel, pad):
        super(SConv_1D, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(6, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        x = self.conv(x)
        return x

numf =12

class Huan_net(nn.Module):
    def __init__(self,num_class = 4):
        super(Huan_net, self).__init__()
        
        input_size = 1
        
        self.DWT0= DWT1DForward(J=1, wave='db16').cuda()
        
        self.SConv1 = SConv_1D(input_size*2, numf, 3, 0)
        self.DWT1= DWT1DForward(J=1, wave='db16').cuda()
        self.dropout1 = nn.Dropout(p=0.1)
        self.cSE1 = A_cSE(numf*2)
        
        self.SConv2 = SConv_1D(numf*2, numf*2, 3, 0)
        self.DWT2= DWT1DForward(J=1, wave='db16').cuda() 
        self.dropout2 = nn.Dropout(p=0.1)
        self.cSE2 = A_cSE(numf*4)
        
        self.SConv3 = SConv_1D(numf*4, numf*4, 3, 0)
        self.DWT3= DWT1DForward(J=1, wave='db16').cuda()       
        self.dropout3 = nn.Dropout(p=0.1)
        self.cSE3 = A_cSE(numf*8)
        
        self.SConv6 = SConv_1D(numf*8, numf*8, 3, 0)              
        
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(numf*8, num_class)

        
    def forward(self, input):
        
        DMT_yl,DMT_yh = self.DWT0(input)
        output = torch.cat([DMT_yl,DMT_yh[0]], dim=1)
        
        output = self.SConv1(output)
        DMT_yl,DMT_yh = self.DWT1(output)
        output = torch.cat([DMT_yl,DMT_yh[0]], dim=1)
        output = self.dropout1(output)
        output = self.cSE1(output)
        
        output = self.SConv2(output)
        DMT_yl,DMT_yh = self.DWT2(output)
        output = torch.cat([DMT_yl,DMT_yh[0]], dim=1) 
        output = self.dropout2(output)
        output = self.cSE2(output)
        
        output = self.SConv3(output)
        DMT_yl,DMT_yh = self.DWT3(output)
        output = torch.cat([DMT_yl,DMT_yh[0]], dim=1) 
        output = self.dropout3(output)
        output = self.cSE3(output)
        
        output = self.SConv6(output)             
            
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        
        return output
#%%  sinc net
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / ((2 * math.pi * band * t_right) + 1e-6)
    y_left = torch.flip(y_right, [0])
    y = torch.cat([y_left, torch.ones(1).to(t_right.device), y_right])
    return y

class SincConv_fast(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels=1):
        super().__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel (here, in_channels = {in_channels})")

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if kernel_size % 2 == 0:
            self.kernel_size += 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):
        half_kernel = self.kernel_size // 2
        time_disc = torch.linspace(-half_kernel, half_kernel, steps=self.kernel_size).to(waveforms.device)
        self.a_ = self.a_.to(waveforms.device)
        self.b_ = self.b_.to(waveforms.device)
        
        filters = []
        for i in range(self.out_channels):
            band = self.a_[i]
            t_right = time_disc - self.b_[i]
            filter = sinc(band, t_right)
            filters.append(filter)

        filters = torch.stack(filters)
        self.filters = filters.view(self.out_channels, 1, -1)

        return F.conv1d(waveforms, self.filters, stride=1, padding=half_kernel, dilation=1, bias=None, groups=1)
    
class SincNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, num_class=4, zero_init_residual=False, Wave_first = True):
        super(SincNet, self).__init__()
        self.inplanes = 64


        
        self.conv1 = SincConv_fast(64, 16)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
 



if __name__ == '__main__':  
    # from torchsummary import summary
    from torchinfo import summary
    from symbolic_layer import DFN
    import argparse
    import yaml
    
    
    # from torch.utils.tensorboard import SummaryWriter
    # save_dir = './save_dir'
    # create_dir(save_dir)
    # summaryWriter = SummaryWriter(save_dir + "/test")
    with open('code/experiment/THU.yaml') as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)
    
    base = symbolic_base(args.symbolic_base_list)
    # self.bases = [base] * len(args.scale) if 1 else AssertionError # 定制不同层的bases ,目前三层都一样,后期怎么打印公式又成了问题
            
    expert_list = [symbolic_base(args.expert_list , fre = args.fre, lenth = args.lenth)]*args.expert_layer if 1 else AssertionError
    feature_list = [symbolic_base(args.feature_list ,device = args.device)] if 1 else AssertionError
    logic_list = [symbolic_base(args.logic_list ,device = args.device)]*args.logic_layer if 1 else AssertionError        
    model_dict = {
        'sinc_net': SincNet(BasicBlock, [2, 2, 2, 2],num_class = 4),
        # 'wave_resnet': ResNet(BasicBlock, [2, 2, 2, 2],num_class = 4),
        # 'resnet': ResNet(BasicBlock, [2, 2, 2, 2],Wave_first=False,num_class = 4),
        # 'wave_resnet_1_4': ResNet_1_4(BasicBlock, [2, 2, 2, 2],num_class = 4),
        # 'resnet_1_4': ResNet_1_4(BasicBlock, [2, 2, 2, 2],Wave_first=False,num_class = 4),
        # 'F-EQL':f_eql(num_class = 4),
        # 'Huan_net':Huan_net(num_class = 4),
        # 'Dong_net':Dong_ELM(num_class = 4),
        # 'DFN':DFN(input_channel = args.input_channel,
        #         bias = args.bias,
        #         symbolic_bases = base,  #meiyongdao
        #         lenth = args.lenth,
        #         scale = args.scale, # feature scale
        #         # sr_scale = self.args.sr_scale # sr scale
        #         skip_connect = args.skip_connect,
        #         down_sampling_kernel = args.down_sampling_kernel,
        #         down_sampling_stride = args.down_sampling_stride,
        #         num_class = args.num_class,
        #         temperature = args.temperature,
        #         expert_list = expert_list,
        #         feature_list = feature_list,
        #         logic_list = logic_list,                
        #         device = args.device,
        #         amount = args.amount)
    }
    import os 
    # os.makedirs('/params', exist_ok=True)
    
    for k,v in model_dict.items():
        net = v
        net_summaary= summary(net.cuda(),(2,1,1024),device = "cuda")
        print(net_summaary)
        with open(f'{k}.txt','w') as f:
            f.write(str(net_summaary))        
    # # wave_res
    # x = torch.randn(32,1,1024).cuda()
    # m_resnet = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()
    # net_summaary= summary(m_resnet,(1,1024),device = "cuda")
    # print(net_summaary)
    # with open('m_resnet.txt','w') as f:
    #     f.write(str(net_summaary))
    # # Res    
    # resnet = ResNet(BasicBlock, [2, 2, 2, 2],Wave_first=False).cuda()
    # net_summaary= summary(m_resnet,(1,1024),device = "cuda")
    # print(net_summaary)
    # with open('resnet.txt','w') as f:
    #     f.write(str(net_summaary))
        
    # # wave_res
    # m_resnet_1_4 = ResNet_1_4(BasicBlock, [2, 2, 2, 2]).cuda()
    # net_summaary= summary(m_resnet_1_4,(1,1024),device = "cuda")
    # print(net_summaary)
    # with open('m_resnet_1_4.txt','w') as f:
    #     f.write(str(net_summaary))
        
    