# common modules

import wandb
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

import sys
sys.path.append("../")
sys.path.append("./code")

# specific modules
from model.symbolic_layer import basic_model,neural_symbolc_base,neural_symbolic_regression,DFN
from model.symbolic_base import symbolic_base
# from model.comparision_net import *
# import data
from data_dir.data_loader import data_default,data_multiple

# from utils
import time
from utils.file_utils import create_dir,loger
from utils.model_utils import get_arity,wgn2
from utils.training_utils import EarlyStopping,clip_gradient,set_seed,metric_recorder,loop_iterable,set_requires_grad,cm_recorder
from utils.training_utils import param_Recorder

from utils.symbolic_utils import get_model_equation
from utils.loss_utils import l1_reg,softmax_entropy,mixuploss




class experiment_basic():

    def __init__(self,args) -> None:
        
        base = symbolic_base(args.symbolic_base_list)
        self.bases = [base] * len(args.scale) if 1 else AssertionError 
        
        create_dir(args.save_dir)
        create_dir(args.plot_dir)
        set_seed(args.seed)
        self.args = args      
                
        self.net = DFN(input_channel = self.args.input_channel,
                bias = self.args.bias,
                symbolic_bases = self.bases,
                lenth = self.args.lenth,
                scale = self.args.scale, # feature scale
                # sr_scale = self.args.sr_scale # sr scale
                skip_connect = self.args.skip_connect,
                down_sampling_kernel = self.args.down_sampling_kernel,
                down_sampling_stride = self.args.down_sampling_stride,
                num_class = self.args.num_class,
                device = self.args.device,
                amount = self.args.amount)
        
        optimizer  = {'Adam':optim.Adam,
                     'sgd':optim.SGD}
        regularzation = {
            'l1':l1_reg,
            'entropy': softmax_entropy
        }


        layers = []
        for i, module in enumerate(self.net.children()):
            # if not isinstance(module, nn.Sequential):
                layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]         
        parameters_conv = ()
        for layer in layers:
            if isinstance(layer, neural_symbolc_base):                
                parameters_conv = *parameters_conv,(layer.channel_conv,"weight") 
                parameters_conv = *parameters_conv,(layer.down_conv,"weight")

        
        self.optimizer = optimizer[self.args.optimizer](
            [{'params':self.net.parameters()},
                {'params':parameters_conv,
             'lr':self.args.conv_lr,
             'weight_decay':self.args.weight_decay}],
                lr=self.args.lr)
        
        self.criterion = nn.CrossEntropyLoss() 
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, patience = int(self.args.patience//4)) 
        # 
        self.regularzation = regularzation[self.args.reg]
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)             
        self.path = self.args.checkpoints + '/' + self.args.setting
        self.amount = args.amount
        
        create_dir(self.path)
                 
        print(f"exp {self.args.setting} begin")
        
    def get_data(self):

        dataset = data_default(self.args.data_dir,flag = 'train')
        train_loader = DataLoader(
            dataset = dataset,
            batch_size= self.args.batch_size,
            shuffle = True,
            num_workers = self.args.num_workers
        )
        dataset = data_default(self.args.data_dir,flag = 'val')
        val_loader = DataLoader(
            dataset = dataset,
            batch_size= self.args.batch_size,
            shuffle = False,
            num_workers = self.args.num_workers
        )
        dataset = data_default(self.args.data_dir,flag = 'test')
        test_loader = DataLoader(
            dataset = dataset,
            batch_size= self.args.batch_size,
            shuffle = False,
            num_workers = self.args.num_workers
        )     
        return train_loader, test_loader, val_loader
        
    
    def run(self):

        train_loader, test_loader, val_loader = self.get_data()
        

        wandb.watch(self.net,log=all) # https://docs.wandb.ai/v/zh-hans/integrations/pytorch
        print("'training start!!'")
        
        lowest_loss = 100 # 根据验证集找测试集合结果

        for epoch in range(self.args.epoches):
            
            print(f'{epoch} ==> {self.args.epoches}')
            
            if epoch % 100 == 0 and self.args.visualize_weight:
                self.visualize_weight(epoch)
            
            epoch_start_time = time.time()
            loss_train, metrics_train = self.train(train_loader)
            training_time = time.time() - epoch_start_time
            
            loss_val, metrics_val = self.evaluation(val_loader)
            val_time = time.time() - epoch_start_time - training_time
            self.early_stopping(loss_train,self.net,self.path) # 一般是 loss_val

                        
            loss_test, metrics_test = self.evaluation(test_loader)
            test_time = time.time() - epoch_start_time - training_time - val_time
            
            print(f'#time# || training time ==> {training_time:.2f} ||, || val time ==> {val_time:.2f} || , || test time ==> {test_time:.2f} ||')
            print(f'#Loss# || training loss ==> {loss_train:.4f} ||, || val loss ==> {loss_val:.4f} || , || test loss ==> {loss_test:.4f} ||')
            
            for met in self.args.metric_list:
                print(f'#{met}# || training {met} ==> {metrics_train[met]:.4f} ||, || val {met} ==> {metrics_val[met]:.4f} || , || test {met} ==> {metrics_test[met]:.4f} ||')

            print('saving logger')
            
            # 单独记录吧
            metrics = {'loss':loss_train,
                       **metrics_train,
                       'valloss':loss_val,
                       'valacc':metrics_val['acc'],
                       'valf1':metrics_val['f1'],
                       'testloss':loss_test,
                       'testacc': metrics_test['acc'],
                       'testf1': metrics_test['f1'],                                              
                       }
            wandb.log(metrics)
            
            if (loss_val < lowest_loss ): # 有BUG 无法自定义summary，自己打印吧
                wandb.run.summary["best_accuracy"] = metrics_test['acc']
                lowest_loss = loss_val          
            
            if self.early_stopping.early_stop:
                print("Early stopping")
                self.early_stopping.load_checkpoint(self.net,self.path) # 不需要返回对象
                
                # loss_test, metrics_test = self.evaluation(test_loader)
                
                self.save_equation()                            
                return wandb.run.summary["best_accuracy"]
        # loss_test, metrics_test = self.evaluation(test_loader)
        self.early_stopping.load_checkpoint(self.net,self.path)
        self.save_equation()
        return wandb.run.summary["best_accuracy"]           

    def save_equation(self):
        # plot cm
        self.cm = cm_recorder(device = self.args.device)
        train_loader, test_loader, val_loader = self.get_data()
        test_x = test_loader.dataset.x.to(self.args.device)
        test_y = test_loader.dataset.y.to(self.args.device)
        y_pre = self.net(test_x)
        self.cm.update(y_pre,test_y)
        np.save(self.path + f'/{self.amount}Confusion matrix.npy',self.cm.output.cpu().numpy())   
        # fine tune     
        if self.args.fine_tune:
            self.fine_tune()
        eq,feature_dic,output_dic = get_model_equation(self.net,
                        input_dim = self.args.input_channel,
                        output_dim = self.args.num_class,
                        save_dir = self.path,
                        amount = self.amount) #取稀疏的结构
        for i,feature in enumerate(feature_dic['sympy']):
            print(f'learned feature f{i} ====>> {feature}')
        for i,output in enumerate(output_dic['sympy']):
            print(f'learned equation f{i} ====>> {output}')            
        # wandb.log(feature_dic)s
        # wandb.log(output_dic)
        # torch.onnx.export(self.net, test_loader.dataset.x.to(self.args.device), "model.onnx")
        # wandb.save("model.onnx")        
                
    def fine_tune(self):
        '''
        todo
        '''
        layers = []
        for i, module in enumerate(self.net.children()):
            if not isinstance(module, nn.Sequential):
                layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]
        for layer in layers:
            if isinstance(layer, neural_symbolc_base):
                prune.l1_unstructured(layer.channel_conv,name="weight",amount = self.amount)                
                prune.l1_unstructured(layer.down_conv,name="weight",amount = self.amount)

        set_requires_grad(self.net,False)
        set_requires_grad(self.net.regression_layer,True)
        train_loader, test_loader, val_loader = self.get_data()
        print('fine-tune')
        self.ftearly_stopping = EarlyStopping(patience=int(self.args.patience * 2), verbose=True,sparse=str(self.amount))  
        for epoch in range(self.args.ftepoches):
            epoch_start_time = time.time()
            loss_train, metrics_train = self.train(train_loader)
            training_time = time.time() - epoch_start_time
            
            loss_val, metrics_val = self.evaluation(val_loader)
            val_time = time.time() - epoch_start_time - training_time
            self.ftearly_stopping(loss_val,self.net,self.path)

                        
            loss_test, metrics_test = self.evaluation(test_loader)
            test_time = time.time() - epoch_start_time - training_time - val_time
            
            metrics = {'ftloss':loss_train,
                        'ftacc':metrics_train['acc'],
                        'ftf1':metrics_train['f1'],
                        'ftvalloss':loss_val,
                        'ftvalacc':metrics_val['acc'],
                        'ftvalf1':metrics_val['f1'],
                        'fttestloss':loss_test,
                        'fttestacc': metrics_test['acc'],
                        'fttestf1': metrics_test['f1'],                                              
                        }
            if self.ftearly_stopping.early_stop:
                print("Early stopping")
                self.ftearly_stopping.load_checkpoint(self.net,self.path) # 不需要返回对象
                
                break
                           
            wandb.log(metrics) # 最后一行
    def visualize_weight(self,epoch):
        weight = {}
        weight['name'] = []
        weight['parms_weight'] = []
        
        for name, parms in self.net.named_parameters():
            weight['name'].append(name if name != None else -1)
            weight['parms_weight'].append(parms.cpu().detach().numpy() if name != None else -1)
        np.save(self.path + f'/epoch{epoch}',weight, allow_pickle= True) # np.load("data.npy", allow_pickle=True).item()    # 读取
        

                                       
    def train(self,loader):
        self.net.train()
        iteration = int(loader.dataset.length//self.args.batch_size)
        loss_per_epoch = []
        # reg_loss_per_epoch = [] 先不log了
        batch_iterator = loop_iterable(loader)        
        for iter in range(iteration):
            batch = next(batch_iterator)                   
            x, y = batch
            x, y = x.to(self.args.device), y.to(self.args.device) 
       
            self.optimizer.zero_grad()
            # forward
            
            regularization_loss = 0           
            for param in self.net.parameters():
                
                # regularization_loss += torch.sum(torch.abs(param))
                regularization_loss += self.regularzation(param = param)
            y_pre = self.net(x)               
            loss = self.criterion(y_pre,torch.max(y, 1)[1])  + self.args.lamda * regularization_loss# criterion 和 metric 分开
            if self.args.mixup:
                loss += mixuploss(x,y,self.criterion,self.net,alpha = self.args.mixup)                                 
            loss.backward()
            
            if self.args.clip_gradient is not None:
                clip_gradient(self.optimizer, self.args.clip_gradient)
            
            loss_per_epoch.append(loss.item())
            self.optimizer.step()
            if iter == 0:
                recoder = metric_recorder(self.args.metric_list,num_classes= self.args.num_class)
            recoder.update(y_pre,y)             
            
        return np.mean(loss_per_epoch), recoder.mean()
    
    def evaluation(self,loader,noise = False):
        self.net.eval()
        iteration = int(loader.dataset.length//self.args.batch_size)
        loss_per_epoch = []
        batch_iterator = loop_iterable(loader)
        
        x = loader.dataset.x.to(self.args.device)
        y = loader.dataset.y.to(self.args.device)

         
        ################ add noise experiment here#############
        if noise != False:           
            N = wgn2(x,noise)
            x = x +N.cuda()
        #####################################################
        # forward
        y_pre = self.net(x)
        loss = self.criterion(y_pre,y) # criterion 和 metric 分开                           
        loss_per_epoch.append(loss.item())
        recoder = metric_recorder(self.args.metric_list,num_classes= self.args.num_class)
            
        recoder.update(y,y_pre)
        return np.mean(loss_per_epoch),recoder.mean()  




class experiment_fine_tune(experiment_basic):
    def __init__(self, args) -> None:
        super().__init__(args)
    def evaluation(self, loader,noise):
        return super().evaluation(loader,noise)
    def run(self):
        return super().run()
    def get_data(self):
        return super().get_data()    
    def save_equation(self):
        # plot cm
        self.cm = cm_recorder(device = self.args.device, num_classes = self.args.num_class)
        train_loader, test_loader, val_loader = self.get_data()
        test_x = test_loader.dataset.x.to(self.args.device)
        test_y = test_loader.dataset.y.to(self.args.device)
        y_pre = self.net(test_x)
        self.cm.update(y_pre,test_y)
        np.save(self.path + f'/Confusion matrix.npy',self.cm.output.cpu().numpy())     
        # eq = get_model_equation(self.net,
        #                 input_dim = self.args.input_channel,
        #                 output_dim = self.args.num_class,
        #                 save_dir = self.path,
        #                 amount = self.sparse) 
        
        # 取稀疏的结构 # 和 basic 不同
        # for i,feature in enumerate(feature_dic['sympy']):
        #     print(f'learned feature f{i} ====>> {feature}')
        # for i,output in enumerate(output_dic['sympy']):
        #     print(f'learned equation f{i} ====>> {output}')
    def pruning(self):
        
        layers = []
        for i, module in enumerate(self.net.children()):
            if not isinstance(module, nn.Sequential):
                layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]         
        parameters_to_prune = ()
        for layer in layers:
            if isinstance(layer, neural_symbolc_base):                
                parameters_to_prune = *parameters_to_prune,(layer.channel_conv,"weight") 
                parameters_to_prune = *parameters_to_prune,(layer.down_conv,"weight")  
                
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.args.amount,
    )   

      
              
        
                            
    def train(self,loader):
        self.net.train()
        iteration = int(loader.dataset.length//self.args.batch_size)
        loss_per_epoch = []
        # reg_loss_per_epoch = [] 先不log了
        batch_iterator = loop_iterable(loader)        
        for iter in range(iteration):
            batch = next(batch_iterator)                   
            x, y = batch
            x, y = x.to(self.args.device), y.to(self.args.device) 

       
            self.optimizer.zero_grad()
            # forward
            
            regularization_loss = 0           
            for i, (name,param) in enumerate(self.net.named_parameters()):
                if 'learnable_param' not in name:
                    regularization_loss += self.regularzation(param = param)
            y_pre = self.net(x)               
            loss = self.criterion(y_pre,torch.max(y, 1)[1])  + self.args.lamda * regularization_loss# criterion 和 metric 分开 todo 多分类 
            if self.args.mixup:
                loss += mixuploss(x,y,self.criterion,self.net,alpha = self.args.mixup)                                 
            loss.backward(retain_graph=True)
            
            if self.args.clip_gradient is not None:

                clip_gradient(self.optimizer, self.args.clip_gradient)
            
            loss_per_epoch.append(loss.item())
            self.optimizer.step()
            if iter == 0:
                recoder = metric_recorder(self.args.metric_list,num_classes= self.args.num_class)
            recoder.update(y_pre,y)             
            
        return np.mean(loss_per_epoch), recoder.mean()
    def run(self):
        
        # 可以放到外面

        train_loader, test_loader, val_loader = self.get_data()
        

        # wandb.watch(self.net,log=all) # https://docs.wandb.ai/v/zh-hans/integrations/pytorch
        wandb.watch(self.net,log='all')
        print("'training start!!'")
        
        self.prune_count = 0 # 记录剪枝次数
        
        best_val_loss = np.inf
        best_acc = 0
        
        self.param_recorder = param_Recorder(self.net)
        
        for epoch in range(self.args.epoches):
            
            self.param_recorder.record(epoch, self.net)
            
            print(f'{epoch} ==> {self.args.epoches}')
            
            if epoch == 0 and self.args.visualize_weight: # 第一次保存
                self.visualize_weight(epoch)
            
            epoch_start_time = time.time()
            loss_train, metrics_train = self.train(train_loader)
            training_time = time.time() - epoch_start_time
            
            loss_val, metrics_val = self.evaluation(val_loader,noise = self.args.noise)
            val_time = time.time() - epoch_start_time - training_time
            self.early_stopping(loss_val,self.net,self.path) # 一般是 loss_val

                        
            loss_test, metrics_test = self.evaluation(test_loader,noise = self.args.noise)
            test_time = time.time() - epoch_start_time - training_time - val_time
             
            print(f'#time# || training time ==> {training_time:.2f} ||, || val time ==> {val_time:.2f} || , || test time ==> {test_time:.2f} ||')
            print(f'#Loss# || training loss ==> {loss_train:.4f} ||, || val loss ==> {loss_val:.4f} || , || test loss ==> {loss_test:.4f} ||')
            
            for met in self.args.metric_list:
                print(f'#{met}# || training {met} ==> {metrics_train[met]:.4f} ||, || val {met} ==> {metrics_val[met]:.4f} || , || test {met} ==> {metrics_test[met]:.4f} ||')

            print('saving logger')
            
            # 单独记录吧
            if best_val_loss > loss_val:
                best_val_loss = loss_val
                best_acc = metrics_test['acc']
            metrics = {'loss':loss_train,
                       **metrics_train,
                       'valloss':loss_val,
                       'valacc':metrics_val['acc'],
                       'valf1':metrics_val['f1'],
                       'testloss':loss_test,
                       'testacc': metrics_test['acc'],
                       'testf1': metrics_test['f1'],
                       'bestacc':best_acc                                              
                       }
            wandb.log(metrics)
            if loss_train == np.nan:
                 self.early_stopping.early_stop = True
            if self.early_stopping.early_stop:
                
                self.param_recorder.save(self.path)
                
                print("Early stopping for purning")
                if self.prune_count < self.args.prune_time: # 还在容许范围内，再进行剪枝
                    self.early_stopping.load_checkpoint(self.net,self.path) # 不需要返回对象 , 读取上一个sparse的pth
                    self.sparse = format((1-self.amount) ** (self.prune_count + 1), '.10f')
                    
                    self.early_stopping.reset(self.sparse) # 重新count down，更新sparse


                    self.visualize_weight(epoch)                
                    self.pruning()
                    self.visualize_weight(epoch+1)
                    if self.prune_count >= 1:
                        self.save_equation()
                    self.prune_count += 1
                    
                    best_val_loss = np.inf     
                                           
                else:
                    self.early_stopping.load_checkpoint(self.net,self.path)
                    self.save_equation()
                    break ## 跳出循环
        # loss_test, metrics_test = self.evaluation(test_loader)
        self.early_stopping.load_checkpoint(self.net,self.path)
        self.save_equation()        

              
class experiment_DFN(experiment_fine_tune):
    def __init__(self, args) -> None:
        # super().__init__(args)
        create_dir(args.save_dir)
        create_dir(args.plot_dir)
        set_seed(args.seed)
        self.args = args      
                       
        optimizer  = {'Adam':optim.Adam,
                     'sgd':optim.SGD}
        regularzation = {
            'l1':l1_reg,
            'entropy': softmax_entropy
        }
        base = symbolic_base(args.symbolic_base_list)
        # self.bases = [base] * len(args.scale) if 1 else AssertionError # 定制不同层的bases ,目前三层都一样,后期怎么打印公式又成了问题
                
        self.expert_list = [symbolic_base(args.expert_list , fre = args.fre, lenth = args.lenth)]*args.expert_layer if 1 else AssertionError
        self.feature_list = [symbolic_base(args.feature_list ,device = args.device)] if 1 else AssertionError
        self.logic_list = [symbolic_base(args.logic_list ,device = args.device)]*args.logic_layer if 1 else AssertionError        
        self.net = DFN(input_channel = self.args.input_channel,
                bias = self.args.bias,
                symbolic_bases = base,  #meiyongdao
                lenth = self.args.lenth,
                scale = self.args.scale, # feature scale
                # sr_scale = self.args.sr_scale # sr scale
                skip_connect = self.args.skip_connect,
                down_sampling_kernel = self.args.down_sampling_kernel,
                down_sampling_stride = self.args.down_sampling_stride,
                num_class = self.args.num_class,
                temperature = self.args.temperature,
                expert_list = self.expert_list,
                feature_list = self.feature_list,
                logic_list = self.logic_list,                
                device = self.args.device,
                
                amount = self.args.amount,
                args = self.args)
                
        symbolic_params = []
        learnable_params = []
        for name, param in self.net.named_parameters():
            if 'learnable' in name:
                learnable_params.append(param)
            else:
                symbolic_params.append(param)
                
        self.optimizer = optimizer[self.args.optimizer](
            [{'params':iter(learnable_params),
              'lr':self.args.lr,},
                {'params':iter(symbolic_params),
             'lr':self.args.conv_lr,
             'weight_decay':self.args.weight_decay}]
            )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, patience = int(self.args.patience//4)) 
        # 
        self.regularzation = regularzation[self.args.reg]
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)             
        self.path = self.args.checkpoints + '/' + self.args.setting
        self.amount = args.amount
        
        
        create_dir(self.path)
                 
        print(f"exp {self.args.setting} begin")

from model.comparision_net import *


class experiment_COM(experiment_fine_tune):
    def __init__(self, args) -> None:
        # super().__init__(args)
        create_dir(args.save_dir)
        create_dir(args.plot_dir)
        set_seed(args.seed)
        self.args = args      
                       
        optimizer  = {'Adam':optim.Adam,
                     'sgd':optim.SGD}
        regularzation = {
            'l1':l1_reg,
            'entropy': softmax_entropy
        }
        model_dict = {
            'wave_resnet': ResNet(BasicBlock, [2, 2, 2, 2],num_class = args.num_class),  # 1 
            'resnet': ResNet(BasicBlock, [2, 2, 2, 2],Wave_first=False,num_class = args.num_class), # 2
            'wave_resnet_1_4': ResNet_1_4(BasicBlock, [2, 2, 2, 2],num_class = args.num_class),
            'resnet_1_4': ResNet_1_4(BasicBlock, [2, 2, 2, 2],Wave_first=False,num_class = args.num_class),
            'F-EQL':f_eql(num_class = args.num_class),
            'Huan_net':Huan_net(num_class = args.num_class), # 3
            'Dong_net':Dong_ELM(num_class = args.num_class), # 4
            'Sincnet':SincNet(BasicBlock, [2, 2, 2, 2],num_class = args.num_class)
        }
        
        self.net = model_dict[args.model]
        self.net = self.net.to(args.device)
                
        symbolic_params = []
        learnable_params = []
        for name, param in self.net.named_parameters():
            if 'learnable' in name:
                learnable_params.append(param)
            else:
                symbolic_params.append(param)
                
        self.optimizer = optimizer[self.args.optimizer](
            [{'params':iter(learnable_params),
              'lr':self.args.lr,},
                {'params':iter(symbolic_params),
             'lr':self.args.conv_lr,
             'weight_decay':self.args.weight_decay}]
            )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, patience = int(self.args.patience//4)) 
        # 
        self.regularzation = regularzation[self.args.reg]
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)             
        self.path = self.args.checkpoints + '/' + self.args.setting
        self.amount = args.amount
        
        
        create_dir(self.path)
                 
        print(f"exp {self.args.setting} begin")
 
                        
class experiment_generalization(experiment_DFN): # for 泛化任务 #experiment_fine_tune


    '''
    for 泛化任务
    '''
    
    def __init__(self, args) -> None:
        super().__init__(args)
    def evaluation(self, loader,noise = None):
        return super().evaluation(loader,noise)
    def run(self):
        return super().run()
    def save_equation(self):
        return super().save_equation()
    def get_data(self):

        # data_generator = {'train':data_multiple, 'test':data_default, 'val':data_multiple}
        dataset = data_multiple(self.args.source_list,flag = 'train')
        train_loader = DataLoader(
            dataset = dataset,
            batch_size= self.args.batch_size,
            shuffle = True,
            num_workers = self.args.num_workers
        )
        
        dataset = data_multiple(self.args.source_list,flag = 'val')
        val_loader = DataLoader(
            dataset = dataset,
            batch_size= self.args.batch_size,
            shuffle = False,
            num_workers = self.args.num_workers
        )
        
        dataset = data_default(self.args.target,flag = 'test')
        test_loader = DataLoader(
            dataset = dataset,
            batch_size= self.args.batch_size,
            shuffle = False,
            num_workers = self.args.num_workers
        )        

        return train_loader, test_loader, val_loader
    
    
if __name__ == '__main__':
    model = experiment_DFN 
    print('1')
    
    