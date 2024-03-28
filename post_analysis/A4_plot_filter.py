import sys
sys.path.append('./')
from model.symbolic_base import wave_filters as filters
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import pandas as pd
from A1_plot_config import configure_matplotlib
configure_matplotlib(style='ieee', font_lang='en')

markers = ['*','o', 'x', 'v', '^', 's']
colors = ['g','c', 'b', 'm']
lines = ['dotted','-','--',':','-.']
# 假设 filters 类和 args 已经定义

def plot_filter_response(filters, f_cs, f_bs, plot_dir='./plot_dir'):
    
    print(f'Plotting filter response...')
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = ['*','o', 'x', 'v', '^', 's']
    colors = ['g','c', 'b', 'm']
    lines = ['dotted','-','--',':','-.']

    for i, f_c in enumerate(f_cs):
        for j, f_b in enumerate(f_bs):
            filters.f_c.data[:, 0, :] = f_c
            filters.f_b.data[:, 0, :] = f_b
            f = filters.filter_generator([1, 1, 1024])
            ax.plot(filters.omega.cpu().detach().numpy()[0, 0, :], f.cpu().detach().numpy()[0, 0, :],
                    linestyle=lines[j], marker=markers[j], color=colors[i],
                    label=f'$f_c$={f_c:.1f}, $f_b$={f_b:.2f}', alpha=0.5)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    fig.legend(loc='upper center', ncol=4, facecolor='gray')
    plt.savefig(f'{plot_dir}/filter_all4.pdf', dpi=512)
    plt.savefig(f'{plot_dir}/filter_all4.png', dpi=512)
    plt.show()
    
def plot_time_domain(filters, f_cs, f_bs, plot_dir='./plot_dir'):
    
    print(f'Plotting time domain response...')
    fig, ax_time = plt.subplots(4, 3, figsize=(8, 10))

    for i, f_c in enumerate(f_cs):
        for j, f_b in enumerate(f_bs):
            filters.f_c.data[:, 0, :] = f_c
            filters.f_b.data[:, 0, :] = f_b
            f = filters.filter_generator([1, 1, 1024])
            time = torch.fft.fftshift(torch.fft.irfft(f), dim=2)
            x = np.linspace(0, 1, time.shape[-1])
            ax_time[i, j].plot(x, time.cpu().detach().numpy()[0, 0, :], linestyle='-', marker=markers[j],
                               color=colors[i], label=f'$f_c$={f_c:.1f}, $f_b$={f_b:.2f}', alpha=0.5)
            ax_time[i, j].set_title(f'$f_c$={f_c:.1f}, $f_b$={f_b:.2f}')
            ax_time[i, j].set_xticks([])
            ax_time[i, j].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/filter_time.pdf', dpi=512)
    plt.savefig(f'{plot_dir}/filter_time.png', dpi=512)
    plt.show()


def plot_filter_evolution(filters, params_path = 'post_analysis/params.csv', scales = 4, layers = 3, plot_dir='./plot_dir'):
    """
    绘制滤波器参数随epoch变化的可视化图。

    Args:
        filters: 滤波器实例。
        params_path (str): 滤波器参数CSV文件路径。
        scales (int): 尺度数量。
        layers (int): 层的数量。
        plot_dir (str): 图表保存目录。
    """
    print(f'Plotting filter evolution from {params_path}...')
    params = pd.read_csv(params_path)
    markers = ['*', 'o', 'x', 'v', '^', 's']
    colors = ['r', 'y', 'g', 'c', 'b', 'm']
    lines = ['dotted', '-', '--', ':', '-.']

    for l in range(layers):
        for s in range(scales):
            fig, ax = plt.subplots(figsize=(10, 8))
            f_c_epochs = params.iloc[:, s * layers + l * 2 + 2]
            f_b_epochs = params.iloc[:, s * layers + l * 2 + 3]

            lenth = len(f_c_epochs)
            cmap = cm.get_cmap('cool')

            for i, (f_c, f_b) in enumerate(zip(f_c_epochs, f_b_epochs)):
                color = cmap(i / lenth)

                filters.f_c.data[:, 0, :] = f_c
                filters.f_b.data[:, 0, :] = f_b

                f = filters.filter_generator([1, 1, 1024])
                ax.plot(filters.omega.cpu().detach().numpy()[0, 0, :], f.cpu().detach().numpy()[0, 0, :],
                        linestyle=lines[1], color=color, alpha=0.8)

            ax.set_xlabel('Frequency (Hz)', fontsize=24)
            ax.set_ylabel('Amplitude', fontsize=24)
            ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=lenth))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Epoch', rotation=270, labelpad=20, fontsize=24)
            cbar.ax.tick_params(labelsize=24)

            plt.savefig(f'{plot_dir}/filters_layer{l}_scale{s}.pdf', transparent=True, dpi=512)
            plt.close()


if __name__ == '__main__':
    class Args:
            def __init__(self):
                    self.f_c_mu = 0.25
                    self.f_c_sigma = 0.1
                    self.f_b_mu = 0.25
                    self.f_b_sigma = 0.1
                    self.device = 'cpu'

    args = Args()
    filter = filters(args = args)  # 假设 filters 类和 args 已经定义
    f_cs = [0.1, 0.2, 0.3, 0.4]
    f_bs = [0.01, 0.03, 0.05]
    plot_filter_response(filter, f_cs, f_bs)
    plot_time_domain(filter, f_cs, f_bs)
    plot_filter_evolution(filter,)


