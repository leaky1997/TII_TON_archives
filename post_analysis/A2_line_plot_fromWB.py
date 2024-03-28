import wandb
import matplotlib.pyplot as plt

# 全局变量定义
markers = ['*', 'o', 'x', 'v', '^', 's']
colors = ['r', 'y', 'g', 'c', 'b', 'm']
lines = ['dotted', '-', '--', ':', '-.']

def get_runs_history(run_paths_dict):
    """
    从W&B API批量获取多个运行的历史数据，输入为一个字典。

    Args:
        run_paths_dict: 一个字典，键为标签，值为W&B路径字符串。
    
    Returns:
        dict: 键为标签，值为对应的历史数据(DataFrame)。
    """
    api = wandb.Api()
    history_dict = {}
    for label, path in run_paths_dict.items():
        print(f'Fetching history for {label}...')
        run = api.run(path)
        history = run.history()
        history_dict[label] = history
    return history_dict


def plot_data(values, label, color, window=5, alpha=0.2):
    """
    绘制数据的折线图和置信区间。

    Args:
        values: 数据点的Pandas Series。
        label: 图例标签。
        color: 线条颜色。
        window: 滑动窗口大小用于平滑。
        alpha: 置信区间的透明度。
    """
    plt.plot(values.rolling(window=window).mean(), label=label, color=color)
    plt.fill_between(values.index,
                     values.rolling(window=window).mean() + values.rolling(window=window).std(),
                     values.rolling(window=window).mean() - values.rolling(window=window).std(),
                     alpha=alpha, color=color)

def line_plot(data_dict, plot_path, window=5, ylim=[0, 2], ylabel='', alpha=0.2, dpi=512):
    """
    绘制折线图和置信区间，并保存到指定路径。

    Args:
        data_dict: 包含多个数据集合的字典。
        plot_path: 图表保存的路径。
        window: 滑动窗口大小。
        ylim: y轴的限制。
        ylabel: y轴的标签。
        alpha: 置信区间的透明度。
        dpi: 图片的DPI。
    """
    for i, (label, values) in enumerate(data_dict.items()):
        print(f'Plotting {label}...')
        plot_data(values, label, colors[i], window, alpha)  # 使用封装的绘图函数

    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    
    # 使用提供的路径参数保存图表
    plt.savefig(f'{plot_path}{ylabel}.pdf', transparent=True, dpi=dpi)
    plt.show()
    plt.close()

def line_analysis(run_paths,plot_path = 'plot_dir/',item = 'acc'):

    # 获取所有运行的历史数据
    history_dict = get_runs_history(run_paths)

    # 例如，如果您想要绘制每个运行的accuracy，请确保数据中有对应的字段
    # 假设每个DataFrame都有一个名为'accuracy'的列
    # 你可能需要根据实际情况调整列名
    acc_data_dict = {label: df[item] for label, df in history_dict.items()}


    # 传递plot_path作为参数
    line_plot(data_dict=acc_data_dict, plot_path=plot_path, window=20, ylim=[0, 1.05], ylabel= item.capitalize(), alpha=0.2, dpi=512)

if __name__ == '__main__':
    # Example usage
    # 从W&B API获取运行的历史数据

    from A1_plot_config import configure_matplotlib
    
    configure_matplotlib(style='ieee', font_lang='en')
    run_paths_dict = {
        'AFN': "/richie_team/THU_FD/runs/bt55p8lw",
        'Dong': "/richie_team/THU_com/runs/21dumwb0",
        'WaveResNet': "/richie_team/THU_com/runs/3iegq40p",
        'Huan': "/richie_team/THU_com/runs/3vfhsyp2",
        'ResNet': "/richie_team/THU_com/runs/23g68x5q"
    }
    line_analysis(run_paths_dict,plot_path = 'plot_dir/',item = 'acc')
    line_analysis(run_paths_dict,plot_path = 'plot_dir/',item = 'testacc')


