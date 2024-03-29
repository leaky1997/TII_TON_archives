import sys
sys.path.append('./')
from A1_plot_config import configure_matplotlib
configure_matplotlib(style='ieee', font_lang='en')



import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Iterable, Callable
from torch import nn, Tensor

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# def visualize_signal(x0, num, path, with_stick=True, shift=2):
#     """
#     可视化单个信号及其频谱。
    
#     Args:
#         x0: 信号数据。
#         num: 信号编号。
#         path: 图像保存路径。
#         with_stick: 是否显示坐标轴标签。
#         shift: 频率数据的截断参数。
#     """
#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 1))
#     time_signal = x0.cpu().detach().numpy().squeeze()
#     length = len(time_signal)
#     fre_signal = np.fft.fft(time_signal)[:length//2]
#     fre_signal = np.abs(fre_signal)

#     t = np.linspace(0, 1, length)
#     w = np.linspace(0, 0.5, length//2-shift)
#     axs[0].plot(t, time_signal, linewidth=0.5, color='darkviolet')
#     axs[1].plot(w, fre_signal[shift:], linewidth=0.5, color='blue') # [:100]
#     for ax in axs:
#         ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
#         ax.xaxis.set_tick_params(labelsize=6)
#         ax.yaxis.set_tick_params(labelsize=6)
#         if with_stick:
#             ax.set_xlabel('Time (s)' if ax == axs[0] else 'Frequency (Hz)', fontsize=12)
#             ax.set_ylabel('Amplitude', fontsize=12)
#         else:
#             ax.set_xticks([])
#             ax.set_yticks([])
    
#     plt.savefig(f'{path}signal{num}{with_stick}.pdf', transparent=True, dpi=512)
#     plt.show()
#     plt.close()

# def visualize_series_signals(features, t, w, num, path, with_stick=True, shift=2):
#     """
#     可视化一系列信号及其频谱。
    
#     Args:
#         features: 提取的特征字典，键为层的名字，值为对应的特征张量。
#         t: 时间轴上的点。
#         w: 频率轴上的点。
#         num: 信号编号。
#         path: 图像保存路径。
#         with_stick: 是否显示坐标轴标签。
#         shift: 频率数据的截断参数。
#     """
#     for name, output in features.items():
#         scales = output.shape[1]    
#         fig, axs = plt.subplots(nrows=scales, ncols=2, figsize=(scales*2, 2))
        
#         if scales == 1:
#             axs = [axs]  # 如果只有一个尺度，确保axs是可迭代的
            
#         for scale, ax in enumerate(axs):
#             time_signal = output[0, scale, :].cpu().detach().numpy().squeeze()
#             length = len(time_signal)
#             fre_signal = np.fft.fft(time_signal)[:length//2]
#             fre_signal = np.abs(fre_signal)
#             ax[0].plot(t, time_signal, linewidth=0.5, color='darkviolet')
#             ax[1].plot(w, fre_signal[shift:], linewidth=0.5, color='blue')
#             for a in ax:
#                 a.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
#                 a.xaxis.set_tick_params(labelsize=6)
#                 a.yaxis.set_tick_params(labelsize=6)
#                 if with_stick and scale == scales-1:
#                     a.set_xlabel('Time (s)' if a == ax[0] else 'Frequency (Hz)', fontsize=12)
#                     a.set_ylabel('Amplitude', fontsize=12)
#                 else:
#                     a.set_xticks([])
#                     a.set_yticks([])
        
#         plt.savefig(f'{path}x{name}signal{num}{with_stick}.pdf', transparent=True, dpi=512)
#         plt.show()
#         plt.close()

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output.cpu().detach().numpy()
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


def plot_tf(output, sample, scales, name, path, with_stick=True, shift=2):
    """
    绘制给定信号及其频谱的时间-频率图。

    Args:
        output: 单个信号或包含多组信号的字典。
        sample: 信号的样本编号。
        scales: 信号处理的尺度数。
        name: 信号的名称，用于保存文件。
        path: 图像保存路径。
        with_stick: 是否在图表上显示坐标轴标签。
        shift: 频谱截断参数。
    """
    # 检查output是单个信号还是信号字典
    if isinstance(output, dict):
        
        for signal_name, signal_data in output.items():
            plot_tf(signal_data, sample, scales, signal_name, path, with_stick, shift)
    else:
        print(f'Plotting {name}...')
        fig, axs = plt.subplots(nrows=scales, ncols=2, figsize=(12, scales * 2))
        if scales == 1:
            axs = axs[None, :]  # 如果只有一个尺度，确保axs是可迭代的
        
        for scale in range(scales):
            time_signal = output[sample, scale, :]
            time_signal = (time_signal - np.mean(time_signal)) / (np.std(time_signal) + 1e-6)
            length = len(time_signal)

            t = np.linspace(0, 1, length)
            w = np.linspace(0, 0.5, length // 2 - shift)

            fre_signal = np.abs(fft(time_signal)[:length // 2])
            axs[scale, 0].plot(t, time_signal, linewidth=0.5, color='darkviolet')
            axs[scale, 1].plot(w, fre_signal[shift:], linewidth=0.5, color='blue')

            for ax in axs[scale]:
                ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
                ax.xaxis.set_tick_params(labelsize=6)
                ax.yaxis.set_tick_params(labelsize=6)
                if with_stick and scale == scales - 1:
                    ax.set_xlabel('Time (s)' if ax == axs[scale, 0] else 'Frequency (Hz)', fontsize=12)
                    ax.set_ylabel('Amplitude', fontsize=12)
                else:
                    ax.set_xticklabels([], fontsize=12)
                    ax.set_xticklabels([], fontsize=12)

        plt.savefig(f'{path}{name}signal.pdf', transparent=True, dpi=512)
        plt.show()
        plt.close()



if __name__ == '__main__':
    # 生成特征提取器
    import os
    from experiment.experiment_FD import experiment_DFN,experiment_COM
    from A3_confusion_plus_noise_task import create_experiment
    plot_dir = './plot_dir/'
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
    args, exp = create_experiment(exp_model_dic, config_dir, 'AFN')
    # exp = exp_model_dic['THU_FD'](config_dir['AFN'])
    exp.net.load_state_dict(torch.load(model_state_paths['AFN']))    
    signal_extractor = FeatureExtractor(exp.net, layers=["symbolic_transform_layer.0",
                                                        "symbolic_transform_layer.1",
                                                        "symbolic_transform_layer.2",
                                                        ])

    _, test_loader, _ = exp.get_data()
    testx, testy = test_loader.dataset.x, test_loader.dataset.y
    
    TON_signal = signal_extractor(testx.cuda())
    
    huan_signal = np.load('post_analysis/7_signal/Huan_signal.npy',allow_pickle=True)
    res_signal = np.load('post_analysis/7_signal/res_signal.npy',allow_pickle=True)
    dong_signal = np.load('post_analysis/7_signal/dong_signal.npy',allow_pickle=True)
    WKN_signal = np.load('post_analysis/7_signal/WKN_signal.npy',allow_pickle=True)
    
    sample = 212
    scales = 4
    shift = 1
    plot_tf(output = huan_signal, sample = sample, scales = scales, name = 'Huan', path = plot_dir, with_stick = True, shift = 2)
    plot_tf(output = res_signal, sample = sample, scales = scales, name = 'Res', path = plot_dir, with_stick = True, shift = 2)
    plot_tf(output = dong_signal, sample = sample, scales = 1, name = 'Dong', path = plot_dir, with_stick = True, shift = 2)
    plot_tf(output = WKN_signal, sample = sample, scales = scales, name = 'WKN', path = plot_dir, with_stick = True, shift = 2)
    plot_tf(output = TON_signal, sample = sample, scales = scales, name = 'signal', path = plot_dir, with_stick = True, shift = 2)
    
    
    