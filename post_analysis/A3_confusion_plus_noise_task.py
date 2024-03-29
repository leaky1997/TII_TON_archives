import matplotlib.pyplot as plt
import seaborn as sns
import os
from A1_plot_config import configure_matplotlib
import torch

import sys
sys.path.append('./')
from experiment.experiment_FD import experiment_DFN,experiment_COM
from utils.file_utils import setup_config    

from torchmetrics import ConfusionMatrix
import numpy as np
import pandas as pd

configure_matplotlib(style='ieee', font_lang='en')



def wgn2(x, snr):
    """
    向信号添加高斯白噪声。
    """
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(x ** 2) / (x.size(0) * x.size(1) * x.size(2))
    npower = xpower / snr
    return torch.randn(x.size()).cuda() * torch.sqrt(npower)

def heatmap_confusion(matrix, x_labels=['1', '2', '3'], y_labels=['1', '2', '3'],
                      x_title='Label', y_title='Prediction', title='', font_type='en',
                      plot_dir='./', name='', cmap='cool', save_type='.pdf'):
    """
    绘制混淆矩阵的热图。
    """

    f, ax = plt.subplots(figsize=(9, 8) if font_type == 'en' else (8, 8))

    sns.heatmap(matrix, cmap=cmap, annot=True, fmt=".0f", linewidths=.5,annot_kws={"size": 16},
                ax=ax, xticklabels=x_labels, yticklabels=y_labels)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig(os.path.join(plot_dir, f'{name}_heatmap{save_type}'), transparent=True, dpi=512)
    plt.close()  # 关闭图形，防止重复显示
    
def plot_confusion_and_gen_noise_data(exp_model_dic, config_dir, model_state_paths, noise_dblist, plot_dir):
    """
    运行实验，绘制混淆矩阵，添加噪声并计算准确率。

    Args:
        exp_model_dic (dict): 模型类型字典，映射实验名称到实验函数。
        config_dir (str): 配置文件的路径。
        model_state_paths (dict): 包含模型键和对应模型状态文件路径的字典。
        noise_dblist (list): 噪声dB级别列表。
        plot_dir (str): 保存结果的目录路径。
    """
    results = {}
    for model_key, state_dict_path in model_state_paths.items():
        args, exp = create_experiment(exp_model_dic, config_dir, model_key)
        args, exp, testx, testy, confusion_matrix = plot_confusion_matrix(plot_dir, model_key, state_dict_path)
        accuracy_list = record_accuracy(noise_dblist, plot_dir, model_key, args, exp, testx, testy)
        
        results[model_key] = {'confusion_matrix': confusion_matrix.cpu().numpy(), 'accuracy_list': accuracy_list}
    accuracy_dicts = {model_key: results[model_key]['accuracy_list'] for model_key in results.keys()}
    
    return results,accuracy_dicts

def record_accuracy(noise_dblist, plot_dir, model_key, args, exp, testx, testy):
    accuracy_list = []
    for noise in noise_dblist:
        testx_noise = testx.cuda() + wgn2(testx, noise)
        pred = exp.net(testx_noise.cuda())
        acc = (pred.argmax(dim=1) == testy.argmax(dim=1).to(args.device)).sum().item() / len(testy)
        accuracy_list.append(acc)

        # 保存混淆矩阵和准确率列表
        
    pd.DataFrame(accuracy_list, columns=['Accuracy']).to_csv(os.path.join(plot_dir, f'{model_key}_accuracy_list.csv'), index=False)
    return accuracy_list




def plot_confusion_matrix(args, exp, plot_dir, model_key, state_dict_path):
    ################### could be update for next version ################

    _, test_loader, _ = exp.get_data()
    testx, testy = test_loader.dataset.x, test_loader.dataset.y
    exp.net.load_state_dict(torch.load(state_dict_path))
    aa = exp.net(testx.cuda())
        ################## could be update for next version ################
        
    metric = ConfusionMatrix(num_classes=args.num_class).to(args.device)
    confusion_matrix = metric(aa.argmax(dim=1), testy.argmax(dim=1).to(args.device))
        # 绘制混淆矩阵热图
    heatmap_confusion(confusion_matrix.cpu().numpy(), plot_dir=plot_dir, name=model_key)
    pd.DataFrame(confusion_matrix.cpu().numpy()).to_csv(os.path.join(plot_dir, f'{model_key}_confusion_matrix.csv'), index=False)
    return args,exp,testx,testy,confusion_matrix

def create_experiment(exp_model_dic, config_dir, model_key):
    if model_key == 'AFN':
        config_dir = config_dir['AFN']
    else:
        config_dir = config_dir['COM']
    args = setup_config(config_dir)
    args.model = model_key
    exp = exp_model_dic[args.exp](args)
    return args,exp


def plot_accuracy_noise_effect(accuracy_dicts, snr_levels, plot_dir='./', file_name='accuracy_vs_snr.pdf'):
    """
    绘制不同模型在加噪声影响下的准确率变化。

    Args:
        accuracy_dicts (dict): 模型名称到其在不同SNR下准确率数组的映射。
        snr_levels (list): SNR水平的列表。
        plot_dir (str): 图像保存路径。
        file_name (str): 保存的文件名。
    """
    colors = ['r','y', 'g','c' , 'b', 'm'] 
    markers = ['*','o', 'x', 'v', '^', 's']
    plt.figure(figsize=(10, 6))
    
    # 遍历字典绘制每个模型的准确率变化曲线
    for model_name, accuracies in accuracy_dicts.items():
        plt.plot(snr_levels, accuracies, label=model_name, linewidth=2,
                 c = colors[list(accuracy_dicts.keys()).index(model_name)],
                 marker=markers[list(accuracy_dicts.keys()).index(model_name)])
    
    plt.xticks(np.arange(len(snr_levels)), snr_levels)
    plt.grid()
    plt.xlabel('SNR(dB)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_dir, file_name), transparent=True, dpi=512)
    plt.close()  # 关闭图形以释放内存


if __name__ == '__main__':
    # Example usage
    # 设置实验模型字典
    
    
    # 噪声dB级别列表
    noise_dblist = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    plot_dir = './plot_dir'
    exp_model_dic = {
        'default': experiment_DFN,
        'THU_FD': experiment_DFN,
        'THU_COM': experiment_COM
    }
    # 根目录路径
    root_dir = '/mnt/k/2_work/TII_TON_archives/'

    # 模型状态文件路径字典，使用os.path.join结合root_dir构建完整路径
    model_state_paths = {
        'Huan_net': os.path.join(root_dir, 'save_dir/THU10_huan_4test_lr0.001_Huan_net2/checkpoint.pth'),
        'wave_resnet': os.path.join(root_dir, 'save_dir/THU10_wave_resnet_4test_lr0.001_wave_resnet2/checkpoint.pth'),
        'Dong_net': os.path.join(root_dir, 'save_dir/THU10_wave_4test_lr0.001_Dong_net/checkpoint.pth'),
        'resnet': os.path.join(root_dir, 'save_dir/THU10_resnet_4test_lr0.001_resnet2/checkpoint.pth'),
        'Sincnet': os.path.join(root_dir, 'save_dir/THU10_sinc_net_4test_lr0.001_Sincnet/checkpoint.pth'),
        'AFN': os.path.join(root_dir, 'save_dir/THU10_wave_revisionbig_AFN/checkpoint.pth')
    }

    # 配置文件的路径，也使用os.path.join结合root_dir构建完整路径
    config_dir = {
        'AFN': os.path.join(root_dir, 'experiment/THU.yaml'),
        'COM': os.path.join(root_dir, 'experiment/THU_COM.yaml'),
    }

    plot_dir = os.path.join(root_dir, plot_dir)
    
# 假设exp_model_dic字典和其他函数已经定义
    results, accuracy_dicts = plot_confusion_and_gen_noise_data(exp_model_dic, config_dir, model_state_paths, noise_dblist, plot_dir)

    # 使用返回的准确率数据绘制准确率变化图
    plot_accuracy_noise_effect(accuracy_dicts, noise_dblist, plot_dir=plot_dir, file_name='accuracy_vs_snr_all_models.pdf')
    print('Done!')
