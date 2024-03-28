
"""
MIT License

Copyright (c) 2022 Qi Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to conditions.

Created on Mon OCT 10 19:49:25 2022
@author: Qi Li(李奇)
@Email: liq22@mails.tsinghua.edu.cn // liq22@tsinghua.org.cn

"""
from argparse import ArgumentParser,Namespace
import os


# import experiment.experiment_FD as experiment_FD
from experiment.experiment_FD import experiment_DFN,experiment_COM

import random
import numpy as np
import pandas as pd

# common modules

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")

# specific modules
from model.symbolic_layer import basic_model


# from utils
from utils.file_utils import create_dir, setup_config,loger
from utils.model_utils import get_arity
from utils.training_utils import EarlyStopping, setup_seed
from sklearn.manifold import TSNE
from utils.plot_utils import plot2DDG
# wandb
import wandb



# main entrypoint
def main(config_dir = 'default.yaml',run_flag = True):
    args = setup_config(config_dir)
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('Args in experiment:')
    print(args)
    print(f'setting is {args.setting}')
    # setup_seed(args.seed)
    exp_model_dic = {'default':experiment_DFN,
                     'THU_FD':experiment_DFN,
                    #  'XJTU_RUL':experiment_DFN_RUL,
                     'THU_COM':experiment_COM
 
                     }
    # self.args = args # 用wandb 的config 
    if args.dryrun:
        os.environ['WANDB_MODE'] = 'offline'
    wandb.login()        
    wandb.init(
            project=args.exp,
            config=args,
            name = args.setting
            )
    args = wandb.config # 如果是sweep 模式，他就会选取value中的值 ，例如：'optimizer': {'values': ['adam', 'sgd']} 选sgd
    exp = exp_model_dic[args.exp](args)
    if run_flag:
        
        
        # exp.early_stopping.load_checkpoint(exp.net,exp.path) # 加载模型
        exp.run()


    torch.cuda.empty_cache()
def main_swap(config_dir = 'default.yaml',run_flag = True,agent_id = False):
    args = setup_config(config_dir)
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('Args in experiment:')
    print(args)
    print(f'setting is {args.setting}')
    # setup_seed(args.seed)
    exp_model_dic = {'default':experiment_DFN,
                     'THU_FD':experiment_DFN,
                     'THU_COM':experiment_COM,
                     'THU_TON_swap4':experiment_DFN,
                     'THU_TON_swap_ini':experiment_DFN,
                     'THU_TON_swap_scale':experiment_DFN,
                     'THU_TON_swap4':experiment_DFN,
 
                     }
    if args.dryrun:
        os.environ['WANDB_MODE'] = 'offline'
    CONFIG = vars(args)   # name_space ==> dic
    sweep_config = {}
    for k, v in CONFIG.items():  # 单独把sweep wandb 的key,value拿出来
        if k in {'method','metric','parameters'}:
            sweep_config.update({k:v}) 
    

    def run(config=None):
        with wandb.init(  #好像会被忽视掉
               config=None,
                name = 'sweep'):
                    CONFIG.update(wandb.config) # 更新选出的config
                    whole_config = Namespace(**CONFIG) # dic ==> name_space
                    experiment_DFN(whole_config).run()
    if agent_id:
        wandb.agent(agent_id, run)
    else:
        sweep_id = wandb.sweep(sweep_config, project=CONFIG['exp'], entity='richie_team')
        wandb.agent(sweep_id, run)                    

    
    torch.cuda.empty_cache()

#%% main  
if __name__ == "__main__":
  
    meta_parser = ArgumentParser(description='symbolic net for PHM')

    meta_parser.add_argument('--config_dir', type=str, default='./code/experiment/THU.yaml', help='THU,XJTU')
    
    # meta_parser.add_argument('--config_dir', type=str, default='./code/experiment/THU_COM.yaml', help='THU,XJTU')
    
    config_dir = meta_parser.parse_args().config_dir
    main(config_dir,run_flag = True)

#%%  sweep
    # meta_parser = ArgumentParser(description='symbolic net for PHM')

    # meta_parser.add_argument('--config_dir', type=str, default='./code/experiment/THUswap.yaml', help='THU,XJTU')
    
    # # meta_parser.add_argument('--config_dir', type=str, default='./code/experiment/THU_COM.yaml', help='THU,XJTU')
    
    
    # config_dir = meta_parser.parse_args().config_dir
    # main_swap(config_dir,run_flag = True,agent_id = False)
#########################################超参数选择##############################################
    # config_dir = './code/experiment/THUswap_ini.yaml'
    # main_swap(config_dir,run_flag = True,agent_id = False)
    # config_dir = './code/experiment/THUswap_scale.yaml'
    # main_swap(config_dir,run_flag = True,agent_id = False)
    # config_dir = './code/experiment/THUswap_lr.yaml'
    # main_swap(config_dir,run_flag = True,agent_id = False)    
    
#%%
#%% sweep
    # config_dir = './experiment/l1_config_sweep.yaml'
    # grid_search_main(config_dir)
    # multiple  sweep
    # grid_search_main_multiple(config_dir,'richie_team/test3/b4ogc5qm')
    # update_config()
