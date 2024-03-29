import sys
sys.path.append('./')
from A1_plot_config import configure_matplotlib
configure_matplotlib(style='ieee', font_lang='en')

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_tsne(feature, labels, plot_style='scatter', path='', save_name='feature_tsne', perplexity=10):
    print(f'Carry t-SNE...')
    norm_feature = (feature - np.mean(feature, axis=0,keepdims= True)) / (np.std(feature, axis=0,keepdims= True))
    if plot_style == 'scatter':
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        tsne_feature = tsne.fit_transform(norm_feature)
    elif plot_style == 'line':
        tsne = TSNE(n_components=1, random_state=0, perplexity=perplexity)
        tsne_feature = tsne.fit_transform(norm_feature)

    # plt.figure(figsize=(8, 6))
    colors = ['darkviolet', 'blue', 'red', 'green']
    markers = ['o', '^', 's', 'd']

    print(f'Plotting {plot_style}...')
    for i, label in enumerate(np.unique(labels)):
        index = labels == label
        if plot_style == 'scatter':
            plt.scatter(tsne_feature[index, 0], tsne_feature[index, 1], c=colors[i], label=label,
                         alpha=0.6, marker=markers[i])
            # plt.plot(tsne_feature[index, 0], tsne_feature[index, 1], markers[i], markersize=2, color=colors[i], label=label)
        elif plot_style == 'line':
            num = len(tsne_feature)
            x = np.arange(1, num+1, 1)
            plt.plot(x[index], tsne_feature[index, 0], markers[i], markersize=2, color=colors[i], label=label)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.legend(fontsize=10)
    plt.savefig(f'{path}/{save_name}.pdf', transparent=True, dpi=512)
    plt.show()
    plt.close() 




def plot_violin(feature, labels,path, save_name, top_k=4):
    """
    根据不同类别绘制特征的小提琴图。 在新版本采用 errorbar绘制。

    Args:
        feature (np.array): 原始特征矩阵，形状为 (samples, features)。
        labels (np.array): 对应于特征的类别标签。
        save_name (str): 文件保存名称，也用于判断是否选择特征。
        top_k (int): 选择方差最大的特征数量。
    """
    # 选择方差最大的 top_k 个特征
    feature = select_top_k_var_features(feature, save_name, top_k)
    # 特征标准化

    plt.figure(figsize=(10, 6))
    
    # 为每个类别绘制小提琴图
    unique_labels = np.unique(labels)
    colors = ['darkviolet', 'blue', 'red', 'green']  # 根据实际类别数量调整

    print(f'Plotting violin plot...')
    for i, label in enumerate(unique_labels):
        idx = labels == label
        sns.violinplot(data=feature[idx], color=colors[i],label=f'{label}',
                       linewidth=1,alpha = 0.1, edgecolor=colors[i],width=1)

    plt.xticks(range(top_k), [f'Feature {i+1}' for i in range(top_k)], fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.legend(unique_labels, fontsize=10)
    plt.savefig(f'{path}/{save_name}_violin.pdf', transparent=True, dpi=512)
    plt.show()
    plt.close()

def select_top_k_var_features(feature, save_name, top_k=4):
    """
    如果满足条件，则选择方差最大的 top_k 个特征。
    """

    print(f'Selecting top {top_k} features...')
    variances = np.var(feature, axis=0)
    topk_indices = np.argsort(variances)[-top_k:]
    feature = feature[:, topk_indices]
    # if 'WKN' in save_name or 'Resnet' in save_name or 'MWA' in save_name:
    feature = (feature - np.mean(feature, axis=0)) / (np.std(feature, axis=0) + 1e-6)
    
    return feature
import numpy as np

def create_labels(feature_shape, num_classes):
    """
    创建一个标签数组，其形状与特征矩阵的行数相同，并将给定数量的类别平均分配给这些标签。

    Args:
        feature_shape (tuple): 特征矩阵的形状，即(feature.shape[0], feature.shape[1])。
        num_classes (int): 要分配的类别数量。

    Returns:
        np.array: 标签数组，每个类别被平均分配给特征矩阵的行。
    """
    num_samples = feature_shape[0]
    labels = np.zeros(num_samples, dtype=int)
    
    # 计算每个类别应该分配的样本数量
    samples_per_class = num_samples // num_classes
    
    for i in range(num_classes):
        # 为每个类别分配样本
        start_index = i * samples_per_class
        if i == num_classes - 1:
            # 确保最后一个类别包含所有剩余的样本
            labels[start_index:] = i
        else:
            end_index = start_index + samples_per_class
            labels[start_index:end_index] = i
            
    return labels



if __name__ == '__main__':
    # 读取特征
    import numpy as np

    # 加载特征
    feature = np.load('post_analysis/6_feature/feature.npy', allow_pickle=True)  # AFN_feature 
    ELM_feature = np.load('post_analysis/6_feature/dong.npy', allow_pickle=True)
    WKN_feature = np.load('post_analysis/6_feature/wkn.npy', allow_pickle=True)
    Resnet_feature = np.load('post_analysis/6_feature/res.npy', allow_pickle=True)
    MWA_CNN_feature = np.load('post_analysis/6_feature/huan.npy', allow_pickle=True)

    num = feature.shape[0]

    # 重新整形并创建标签
    labels = create_labels(feature.shape, num_classes=4)

    features_dict = {
        'AFN': feature.reshape(num, -1),
        'ELM': ELM_feature.reshape(num, -1),
        'WKN': WKN_feature.reshape(num, -1),
        'Resnet': Resnet_feature.reshape(num, -1),
        'MWA_CNN': MWA_CNN_feature.reshape(num, -1),
}

    for model_name, feature_set in features_dict.items():
        print(f'Processing {model_name} features...')
        plot_tsne(feature_set, labels, plot_style='scatter', path='./plot_dir', save_name=f'{model_name}_feature_tsne2', perplexity=10)
        plot_tsne(feature_set, labels, plot_style='line', path='./plot_dir', save_name=f'{model_name}_feature_tsne1', perplexity=10)
        plot_violin(feature_set, labels,path='./plot_dir', save_name=f'{model_name}_feature_violin', top_k=4)
