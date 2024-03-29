# from sympy import *
import sympy
from sympy import Function
import math
import torch # .nn as nn
import torch.nn as nn
import torch.nn.functional as F
from sympy.stats import Normal,density
from sympy.abc import W


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
    return torch.fft.ifft(Xf).abs()

class frequency_operation(nn.Module):
    '''
    conference version 
    '''
    def __init__(self, mu=0.1, sigma=1,
                 length=1024, fs = 25600,
                 type = 'fre_normal',device = 'cuda'
                 ) -> None:
        super().__init__()
        
        self.mu = nn.Parameter(torch.Tensor([mu])) #.to(device) 
        self.sigma = nn.Parameter(torch.Tensor([sigma])) #.to(device) 
        self.length = length//2 if length//2 == 0 else length//2 +1
        self.x = torch.linspace(0, 0.5, self.length).to(device)    # 0-0.5的归一化频率范围
        self.type = type
        self.device = device
        
        nn.init.normal_(self.mu,std=0.1)
        nn.init.normal_(self.sigma,std=0.1)
        

        self.to(device)

    def normal(self):
        shift = self.x - self.mu.to(self.device) 
        normal = torch.exp(-0.5 * ((shift) / self.sigma) ** 2) / (self.sigma * math.sqrt(2 * torch.pi))
        return normal
    
    def sinc(self):
        shift = self.x/self.sigma - self.mu.to(self.device) 
        sinc = torch.sin(shift)/(shift+1e-12)     
        return sinc  
    
    def forward(self,x):
        # b,c,l
        c,lenth,device = *x.shape[-2:],x.device
        x = torch.fft.rfft(x, dim=2, norm='ortho') # 

        if self.type == 'fre_normal':
            self.weight = self.normal()
        elif self.type == 'fre_sinc':
            self.weight = self.sinc()
        self.weight = self.weight.to(device)

        x =  x * self.weight
        x = torch.fft.irfft(x, dim=2, norm='ortho')
        x = x.view(-1, c, lenth).real
        return x
    
class adaptive_shrink(nn.Module):
    def __init__(self,init_a,init_b) -> None:
        super().__init__()
        
        self.alpha = nn.Parameter(torch.ones(1))*init_a
        self.b_left = nn.Parameter(torch.ones(1))*init_b
        self.b_right = nn.Parameter(torch.ones(1))*init_b
        
    def forward(self, x):
        
        left = self.alpha.cuda() * (x + self.b_left.cuda())
        right = - self.alpha.cuda() * (x - self.b_right.cuda())
        out = x * (F.sigmoid(left) + F.sigmoid(right))
        return out
    
class wave_filters(nn.Module):
    
    '''
    TII version
    '''
    def __init__(self, in_channels=4, signal_length=1024, args=None):
        super().__init__()
        self.arity = 1 # 输入信号个数 这个版本默认都是1  ，如果输入2输出copy1份也是2比较方便
        self.args = args
        self.device = args.device
        self.to('cuda')
        f_c = torch.rand(1, in_channels, 1,dtype=torch.float,device=self.device) # 归一化频率
        f_b = torch.rand(1, in_channels, 1,dtype=torch.float,device=self.device) # 归一化频率
        
        self.f_c = nn.Parameter(f_c)# .to(self.device)    # center frequency  # don't use to because it create a new tensorhttps://discuss.pytorch.org/t/typeerror-cannot-assign-torch-cuda-floattensor-as-parameter-weight-torch-nn-parameter-or-none-expected/61765
        self.f_b = nn.Parameter(f_b)# .to(self.device)    # band width

        nn.init.normal_(self.f_c, mean=args.f_c_mu, std=args.f_c_sigma)
        nn.init.normal_(self.f_b,  mean=args.f_b_mu, std=args.f_b_sigma)
        self.register_parameter(f'learnable_param_f_c', self.f_c)   
        self.register_parameter(f'learnable_param_f_b', self.f_b)       
        

        
    def filter_generator(self,shape):
        '''
        
        $e^{-(\omega-2\pi f_c)^{2}/(4f_b ^{2})$
        
        '''
        batch_size,channel,length = shape
        fre_lenth = length//2 if length//2 == 0 else length//2 +1
        
        
        self.omega = torch.linspace(0, 0.5, fre_lenth).to(self.args.device) # 0-0.5的归一化频率范围

        self.omega = self.omega.reshape(1, 1, fre_lenth).repeat([1, channel, 1])
        
        #  calculate filters
        
        shift =  self.omega - self.f_c # 2 * torch.pi * self.f_c    
        band_term = 2 * self.f_b        
        filter = torch.exp(-(shift / band_term) ** 2)
        
        return filter
        
    def forward(self,x):
        
        batch_size, channel,lenth = x.shape
        
        self.filter = self.filter_generator(x.shape)
        
        fre = torch.fft.rfft(x, dim=2, norm='ortho') # 

        fre =  fre * self.filter
        
        
        x_hat = torch.fft.irfft(fre, dim=2, norm='ortho')
        
        x_hat = x_hat.view(batch_size, channel, lenth).real
        
        return x_hat
                
class frequency_global_operation(nn.Module):
    def __init__(self,
                 length=1024, fs = 25600,
                 type = 'fre_normal',device = 'cuda'
                 ) -> None:
        super().__init__()
        
        self.length = length//2 if length//2 == 0 else length//2 +1
        
        self.x = torch.linspace(0, 0.5, self.length).to(device)    # 0-0.5的归一化频率范围
        self.type = type
        self.device = device
        
        

        self.to(device)
        self.set_weight()
        
    def set_weight(self):
        self.weight = nn.Parameter(torch.randn(self.length, dtype=torch.float32))
        
    
    # def set_attention(self):
    
    def forward(self,x):
        # b,c,l
        c,lenth,device = *x.shape[-2:],x.device
        x = torch.fft.rfft(x, dim=2, norm='ortho') # 
        x =  x * self.weight.softmax(dim=-1)
        x = torch.fft.irfft(x, dim=2, norm='ortho')
        x = x.view(-1, c, lenth).real
        return x     
    
KERNEL_SIZE = 49 
FRE = 10 
DEVICE = 'cuda'
STRIDE = 1
T = torch.linspace(-KERNEL_SIZE/2,KERNEL_SIZE/2, KERNEL_SIZE).view(1,1,KERNEL_SIZE).to(DEVICE) # 暂定cuda 

class Swish(nn.Module):
	def __init(self,inplace=True):
		super(Swish,self).__init__()
		self.inplace=inplace
	def forward(self,x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x*torch.sigmoid(x)

def Morlet(t):
    C = pow(math.pi, 0.25)
    f = FRE
    w = 2 * math.pi * f    
    y = C * torch.exp(-torch.pow(t, 2) / 2) * torch.cos(w * t)
    return y

def Laplace(t):
    a = 0.08
    ep = 0.03
    tal = 0.1
    f = FRE
    w = 2 * math.pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = a * torch.exp((-ep / (torch.sqrt(q))) * (w * (t - tal))) * (-torch.sin(w * (t - tal)))
    return y

class convlutional_operator(nn.Module):
    '''
    conv_sin
    conv_exp
    Morlet
    Laplace
    '''
    def __init__(self, kernel_op = 'conv_sin',
                 dim = 1, stride = STRIDE,
                 kernel_size = KERNEL_SIZE,device = 'cuda',
                 ) -> None:
        super().__init__()
        # length = x.shape[-1]
        self.affline = nn.InstanceNorm1d(num_features=dim,affine=True).to(device)
        op_dic = {'conv_sin':torch.sin,
                  'conv_sin2':lambda x: torch.sin(x**2),
                  'conv_exp':torch.exp,
                  'conv_exp2':lambda x: torch.exp(x**2),
                  'Morlet':Morlet,
                  'Laplace':Laplace}
        self.op = op_dic[kernel_op]
        self.stride = stride
        self.t = torch.linspace(-math.pi/2,math.pi/2, kernel_size).view(1,1,kernel_size).to(device) # 暂定cuda 
        self.kernel_size = kernel_size
        
    def forward(self,x):

        self.aff_t = self.affline(self.t)
        self.weight = self.op(self.aff_t)
        conv = F.conv1d(x,self.weight, stride=self.stride, padding=(self.kernel_size-1)//2, dilation=1, groups=1)# todo add stride B,Cout,L-K+1        
        return conv

class signal_filter_(nn.Module):
    '''
    order1_MA
    order2_MA
    order1_DF
    order2_DF
    '''
    def __init__(self, kernel_op = 'order1_MA',
                 dim = 1, stride = STRIDE,
                 kernel_size = KERNEL_SIZE,device = 'cuda',
                 ) -> None:
        super().__init__()
        # length = x.shape[-1]
        self.affline = nn.InstanceNorm1d(num_features=dim,affine=True).to(device)
        op_dic = {'order1_MA':torch.Tensor([0.5,0,0.5]), #
                  'order2_MA':torch.Tensor([1/3,1/3,1/3]),
                  'order1_DF':torch.Tensor([-1,0,1]),
                  'order2_DF':torch.Tensor([-1,2,-1])}
        self.weight = op_dic[kernel_op].view(1,1,-1).cuda()
        self.stride = stride
        self.t = torch.linspace(-math.pi/2,math.pi/2, kernel_size).view(1,1,kernel_size).to(device) # 暂定cuda 
        self.kernel_size = 3
        
    def forward(self,x):
        # length = x.shape[-1]
        # t = self.affline(self.t)
        conv = F.conv1d(x, self.weight, stride=self.stride, padding=(self.kernel_size-1)//2, dilation=1, groups=1)# todo add stride B,Cout,L-K+1        
        return conv    

def self_conv(x,y):
    batch = x.shape[0]
    length = x.shape[-1]
    out = torch.zeros_like(x)
    for i in range(batch):
        
        x_ = x[i,:,:].unsqueeze(0)
        weight =  y[i,:,:].unsqueeze(0)
        conv = F.conv1d(x_, weight, stride=1, padding=length//2, dilation=1, groups=1)# todo add stride B,Cout,L-K+1
        out[i,:,:] = conv[:,:,:length]
    return out
ONE = torch.Tensor([1]).cuda()
ZERO = torch.Tensor([0]).cuda()

def generalized_softmax(x,y, alpha = 20):
    numerator = x * torch.exp(alpha * x) + y * torch.exp(alpha * y)
    denominator = torch.exp(alpha * x) + torch.exp(alpha * y)
    return numerator / denominator 
def generalized_softmin(x,y, alpha = 20):
    return -generalized_softmax(-x,-y, alpha = alpha)

def implication(x, y):
    return generalized_softmin(ONE, ONE - x + y)

def equivalence(x, y):
    return ONE - torch.abs(x - y)

def negation(x):
    return ONE - x

def weak_conjunction(x, y):
    return generalized_softmin(x, y)

def weak_disjunction(x, y):
    return generalized_softmax(x, y)

def strong_conjunction(x, y):
    return generalized_softmax(ZERO, x + y - 1)

def strong_disjunction(x, y):
    return generalized_softmin(ONE, x + y)

#

def symbolic_base( given_set = ['add','mul','sin','exp','idt','sig'],
                fre = FRE,
                fs = 25600, time = 1,
                stride = 1,
                kernel_size = 3,
                device = 'cuda',
                mu = 0.1,
                sigma = 0.1,
                lenth = 1024
                  ):  # 
    """
    constant 
    """
    conv = ['conv',]
    
    add = ['add', (lambda x, y: x + y), (lambda x, y: x + y), '$+$']
    mul = ['mul', (lambda x, y: x * y), (lambda x, y: x * y), '$\\times$']
    div = ['div', (lambda x, y: x / (y + 1e12)), (lambda x, y: x / y), '$\div$']
    squ = ['squ', (lambda x: x**2), (lambda x: x**2),'$x^2$']
    sin = ['sin', (lambda x: torch.sin( fre * x)), (lambda x: sympy.sin(x)), '$\sin$']
    arcsin = ['arcsin', (lambda x: torch.arcsin( fre * x)), (lambda x: sympy.asin(x)), '$\\asin$']
    idt = ['idt', (lambda x: x), (lambda x: x), '$\mathbf{I}$']
    sig = ['sig', lambda x: torch.sigmoid(fre * x), lambda x: 1 / (1 + sympy.exp(x)), "$\sigma$"]
    X = Normal("X", 0, 1)
    exp = ['exp',lambda x: torch.exp( - fre* (x) **2),lambda x: density(X)(x) * sympy.sqrt(2*sympy.pi), "$e^{\left(-\\frac{x^{2}}{2}\\right)$" ]
    log = ['log',lambda x: torch.log(x),lambda x: sympy.log(x), "$\log$"]
    tanh = ['tanh',lambda x: torch.tanh(x),lambda x: sympy.tanh(x), "$\tanh$"]
    swish = ['swish',(lambda x: Swish()(x)), (lambda x: sympy.Function('conv_exp2')(x))]
    pi             = ['pi',lambda x: x*math.pi,  lambda x: x*sympy.pi, "$\pi$"]
    e              = ['e',lambda x: x*math.e,  lambda x: x*sympy.E, "$e$"]
    
    
    fft            = ['fft',lambda x: torch.abs(torch.fft.fft(x)),  lambda x: sympy.Function('fft')(x), "$fft$"]
    conv      = ['conv',(lambda x,y: self_conv(x,y)), (lambda x,y: sympy.Function('conv')(x,y)), "$'conv'$"]

    # signal processing
    
    conv_sin      = ['conv_sin',convlutional_operator('conv_sin'), (lambda x: sympy.Function('conv_sin')(x)), "$fft$"]
    conv_sin2      = ['conv_sin2',convlutional_operator('conv_sin2'), (lambda x: sympy.Function('conv_sin2')(x)), "$fft$"]
    conv_exp      = ['conv_exp',convlutional_operator('conv_exp'), (lambda x: sympy.Function('conv_exp')(x)), "$fft$"]
    conv_exp2      = ['conv_exp2',convlutional_operator('conv_exp2'), (lambda x: sympy.Function('conv_exp2')(x)), "$fft$"]

    Morlet      = ['Morlet',convlutional_operator('Morlet'), (lambda x: sympy.Function('Morlet')(x)), "$fft$"]
    Laplace      = ['Laplace',convlutional_operator('Laplace'), (lambda x: sympy.Function('Laplace')(x)), "$fft$"]
        
    order1_MA      = ['order1_MA',(lambda x: signal_filter_('order1_MA')(x)), (lambda x: sympy.Function('MA_1')(x)), "$fft$"]
    order2_MA      = ['order2_MA',(lambda x: signal_filter_('order2_MA')(x)), (lambda x: sympy.Function('MA_2')(x)), "$fft$"]

    order1_DF      = ['order1_DF',(lambda x: signal_filter_('order1_DF')(x)), (lambda x: sympy.Function('DF_1')(x)), "$fft$"]
    order2_DF      = ['order2_DF',(lambda x: signal_filter_('order2_DF')(x)), (lambda x: sympy.Function('DF_2')(x)), "$fft$"]
    
    
    envelope = ['envelope',(lambda x: env_hilbert(x)), (lambda x: sympy.Function('env_hilbert')(x))]
    # fre_normal_ = frequency_operation(mu = mu, sigma = sigma, length = lenth, type = 'normal',device = device)
    fre_normal = ['fre_normal',frequency_operation, (lambda x: sympy.Function('fre_normal')(x))]
    # fre_sinc_ = frequency_operation(mu = mu, sigma = sigma, length = lenth, type = 'sinc',device = device)
    fre_sinc = ['fre_sinc',frequency_operation, (lambda x: sympy.Function('fre_sinc')(x))]
    fre_global_weight = ['fre_global_weight',frequency_global_operation, (lambda x: sympy.Function('fre_sinc')(x))]    
    morlet_wave = ['morlet_wave',wave_filters, (lambda x: sympy.Function('morlet')(x))]
    
    # statistic operation
    
    mean = ['mean',(lambda x: torch.mean(x,dim=-1, keepdim = True)), (lambda x: sympy.Function('mean')(x))]
    std = ['std',(lambda x: torch.std(x,dim=-1, keepdim = True)), (lambda x: sympy.Function('std')(x))]
    var = ['var',(lambda x: torch.var(x,dim=-1, keepdim = True)), (lambda x: sympy.Function('var')(x))]
    entropy = ['entropy',(lambda x: (x * torch.log(torch.softmax(x , dim=-1))).mean(dim=-1, keepdim = True)), (lambda x: sympy.Function('entropy')(x))]
    # entropy = ['entropy',(lambda x: (x * torch.log(x)).mean(dim=-1, keepdim = True)), (lambda x: sympy.Function('entropy')(x))]


    max = ['max', (lambda x: torch.max(x)), (lambda x: sympy.Function('max')(x))]
    min = ['min', (lambda x: torch.min(x)), (lambda x: sympy.Function('min')(x))]
    abs_mean = ['abs_mean', (lambda x: torch.mean(torch.abs(x))), (lambda x: sympy.Function('abs_mean')(x))]
    std = ['std', (lambda x: torch.std(x,dim=-1, keepdim = True)), (lambda x: sympy.Function('std')(x))]
    rms = ['rms', (lambda x: torch.sqrt(torch.mean(x ** 2,dim=-1, keepdim = True))), (lambda x: sympy.Function('rms')(x))]
    var = ['var', (lambda x: torch.var(x,dim=-1, keepdim = True)), (lambda x: sympy.Function('var')(x))]
    crest_factor = ['crest_factor', (lambda x: torch.max(x) / torch.sqrt(torch.mean(x ** 2))), (lambda x: sympy.Function('crest_factor')(x))]
    
    clearance_factor = ['clearance_factor', (lambda x: torch.max(x) / torch.mean(torch.abs(x))), (lambda x: sympy.Function('clearance_factor')(x))]
    kurtosis = ['kurtosis',
                (lambda x: (((x - torch.mean(x,dim=-1, keepdim = True)) ** 4).mean(dim=-1, keepdim = True)) / ((torch.var(x,dim=-1, keepdim = True) ** 2).mean(dim=-1, keepdim = True))),
                (lambda x: sympy.Function('kurtosis')(x))]
    
    skewness = ['skewness', (lambda x: ((x - torch.mean(x)) ** 3).mean(dim=-1, keepdim = True) / (torch.std(x) ** 3)), (lambda x: sympy.Function('skewness')(x))]
   
    shape_factor = ['shape_factor', (lambda x: torch.sqrt(torch.mean(x ** 2)) / torch.mean(torch.abs(x))), (lambda x: sympy.Function('shape_factor')(x))]
    crest_factor_delta = ['crest_factor_delta', (lambda x: torch.sqrt(torch.mean(torch.pow(x[1:] - x[:-1], 2))) / torch.mean(torch.abs(x))), (lambda x: sympy.Function('crest_factor_delta')(x))]
    kurtosis_delta = ['kurtosis_delta', (lambda x: ((x[1:] - x[:-1] - torch.mean(x[1:] - x[:-1])) ** 4).mean() / (((x[1:] - x[:-1] - torch.mean(x[1:] - x[:-1])) ** 2).mean()) ** 2), (lambda x: sympy.Function('kurtosis_delta')(x))]


    
        
    # logical rule
    
    imp = ['imp',(lambda x,y: implication(x,y)), (lambda x,y: sympy.Function('F_imp')(x,y))]  # F_→
    equ = ['equ',(lambda x,y: equivalence(x,y)), (lambda x,y: sympy.Function('F_equ')(x,y))] #F_↔
    neg = ['neg',(lambda x: negation(x)), (lambda x: sympy.Function('F_neg')(x))] # F_¬           
    conj = ['conj',(lambda x,y: weak_conjunction(x,y)), (lambda x,y: sympy.Function('F_conj')(x,y))] # 'F_∧'
    disj = ['disj',(lambda x,y: weak_disjunction(x,y)), (lambda x,y: sympy.Function('F_disj')(x,y))] # 'F_∨'
    sconj = ['sconj',(lambda x,y: strong_conjunction(x,y)), (lambda x,y: sympy.Function('F_sconj')(x,y))] # 'F_⨂'
    sdisj = ['sdisj',(lambda x,y: strong_disjunction(x,y)), (lambda x,y: sympy.Function('F_sdisj')(x,y))] # 'F_⨁'
    
    total_set = {
        
        ### basic
        'add':add,'mul':mul,'div':div,'sin':sin,'arcsin':arcsin,'idt':idt,'sig':sig,'exp':exp,'log':log,'tanh':tanh,'swish':swish,
        'pi':pi,'e':e,
        ### signal processing
        'fft':fft,'conv':conv,'conv_sin':conv_sin,'conv_sin2':conv_sin2,'conv_exp':conv_exp,'conv_exp2':conv_exp2,'squ':squ,'Morlet':Morlet,'Laplace':Laplace,
        'envelope':envelope,'fre_normal':fre_normal,'fre_sinc':fre_sinc,'fre_global_weight':fre_global_weight,
        
        'morlet_wave':morlet_wave,
        
        ### feature ###
        'order1_MA':order1_MA,'order2_MA':order2_MA,'order1_DF':order1_DF,'order2_DF':order2_DF,
        
        'mean':mean, 'max':max,'min':min,'abs_mean':abs_mean,'std':std,'rms':rms,'var':var,
        'crest_factor':crest_factor,'clearance_factor':clearance_factor,'kurtosis':kurtosis,
        'skewness':skewness,'shape_factor':shape_factor,'crest_factor_delta':crest_factor_delta,
        'kurtosis_delta':kurtosis_delta,
        'entropy':entropy,
        
        ### logical rule
        'imp':imp,'equ':equ,'neg':neg,'conj':conj,'disj':disj,'sconj':sconj,'sdisj':sdisj
    
    }
    
    torch_base = { fuc: total_set[fuc][1] for fuc in given_set}
    sympy_base = { fuc: total_set[fuc][2] for fuc in given_set}
    # latex_base = { fuc: total_set[fuc][3] for fuc in given_set}
    
    return {
        'torch':torch_base,
        'sympy':sympy_base,
    }
    
    
def feature_base():
    '''
    for f-eql
    '''
    ori_order1_2 = ["OM1_2O",(lambda x : torch.sqrt(torch.abs(x))),
                        (lambda x: Function('OM1_2O')(x)),
                        '$OM1_2O$']
    ori_order2_2 = ["OM2_2O",(lambda x : torch.sqrt(torch.pow(x, 2))),
                        (lambda x: Function('OM2_2O')(x)),
                        '$OM2_2O$']
    ori_order1 = ["OM1O",(lambda x : x),
                    (lambda x: Function('OM1O')(x)),
                    '$OM1O$']
    ori_order2 = ["OM2O",(lambda x :  torch.pow(x, 2)),
                        (lambda x: Function('OM2O')(x)),
                        '$OM2O$']
    ori_order3 = ["OM3O",(lambda x :  torch.pow(x, 3)),
                        (lambda x: Function('OM3O')(x)),
                        '$OM3O$']
    ori_order4 = ["OM4O",(lambda x :  torch.pow(x, 4)),
                        (lambda x: Function('OM4O')(x)),
                        '$OM4O$']

    cen_order1_2 = ["CM1_2O",(lambda x : torch.sqrt(torch.abs(x-torch.mean(x)))),
                        (lambda x: Function('CM1_2O')(x)),
                        '$CM1_2O$']
    cen_order2_2 = ["CM2_2O",(lambda x : torch.sqrt(torch.pow(x-torch.mean(x), 2))),
                        (lambda x: Function('CM2_2O')(x)),
                        '$CM2_2O$']
    cen_order1 = ["CM1O",(lambda x : x-torch.mean(x)),
                        (lambda x: Function('CM1O')(x)), '$CM1O$']
    cen_order2 = ["CM2O",(lambda x : torch.pow(x-torch.mean(x), 2)),
                        (lambda x: Function('CM2O')(x)), '$CM2O$']
    cen_order3 = ["CM3O",(lambda x : torch.pow(x-torch.mean(x), 3)),
                        (lambda x: Function('CM3O')(x)), '$CM3O$']
    cen_order4 = ["CM4O",(lambda x : torch.pow(x-torch.mean(x), 4)),
                        (lambda x: Function('CM4O')(x)), '$CM4O$']
    total_set = {
        'ori_order1_2':ori_order1_2,
        'ori_order2_2':ori_order2_2,
        'ori_order1':ori_order1,
        'ori_order2':ori_order2,
        'ori_order3':ori_order3,
        'ori_order4':ori_order4,
        'cen_order1_2':cen_order1_2,
        'cen_order2_2':cen_order2_2,
        'cen_order1':cen_order1,
        'cen_order2':cen_order2,
        'cen_order3':cen_order3,
        'cen_order4':cen_order4,        
    }
    
    torch_base = { fuc: total_set[fuc][1] for fuc in total_set}
    sympy_base = { fuc: total_set[fuc][2] for fuc in total_set}

    
    return {
        'torch':torch_base,
        'sympy':sympy_base,
 
    }     

         


#%%   
if __name__ == '__main__':
    base = symbolic_base(['fft','squ','Morlet','Laplace','order1_MA','order2_MA','order1_DF','order2_DF'])
    
    signal_processing = symbolic_base(['envelope','fre_normal','fre_sinc'])
    statistics = symbolic_base(['mean','entropy','kurtosis'])
    
    print(base)
    print(signal_processing)
    print(statistics)
        