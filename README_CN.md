# TON_TII 


## 项目简介

TON_TII 是一个旨在构建完全可解释的诊断模型的项目。该项目通过深度学习和符号回归技术，提供了一种新颖的方法来理解和解释模型的决策过程。这种方法特别适用于需要高度透明性和解释性的应用场景，如医疗诊断、故障检测等

## 环境配置

该项目使用 `conda` 管理依赖。请确保您已安装 [Anaconda](https://www.anaconda.com/products/individual) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)，然后运行以下命令来创建并激活一个包含所有依赖的环境：

```bash
conda create --name ton_tii_env --file requirements.txt
```
大部分环境不是必备的,
pytorch

## 如何运行

```bash
python main.py experiment/THU.yaml
```

## 程序结构 TDOO fix bugs
- data_dir/: 包含数据加载器和数据文件。
- experiment/: 包含实验配置文件和实验脚本。
- figs/: 存放生成的图像。
- model/: 包含模型定义文件。
- plot_dir/: 存放生成的绘图文件。
- post_analysis/: 包含后处理和分析脚本。
- reference/: 包含参考文献或相关资料。
- save_dir/: 存放训练好的模型和其他持久化结果。
- utils/: 包含实用工具和辅助函数。
- main.py: 项目的主入口脚本。

TON_TII/
├── data_dir/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── data_loader.py
│   └── Thu10_down.mat
├── experiment/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── experiment_FD.py
│   ├── THU_COM.yaml
│   ├── THU.yaml
│   ├── THUswap_ini.yaml
│   ├── THUswap_lr.yaml
│   └── THUswap_scale.yaml
├── figs/
├── model/
│   ├── __pycache__/
│   ├── comparision_net.py
│   ├── symbolic_base.py
│   ├── symbolic_layer.py
│   ├── test.ipynb
├── plot_dir/
├── post_analysis/
│   ├── __pycache__/
│   ├── 6_feature/
│   ├── 7_signal/
│   ├── __init__.py
│   ├── A1_plot_config.py
│   ├── A2_line_plot_fromWB.py
│   ├── A3_confusion_plus_noise_task.py
│   ├── A4_plot_filter.py
│   ├── A5_signal_vis.py
│   ├── A6_features.py
│   ├── A7_hyperparameter.py
│   ├── analysis_conference_version.ipynb
│   ├── analysis_TII_archive.ipynb
│   ├── analysis_TII_version.ipynb
│   ├── init.csv
│   ├── lr.csv
│   ├── params_trainedby0.00001.csv
│   ├── params.csv
│   ├── scale.csv
│   └── test.ipynb
├── reference/
├── save_dir/
├── utils/
└── main.py


## geiBUG
main.py
- 主程序入口,可以执行THU 实验 THU_COM的对比实验 以及 注释中的sweep 超参数选择实验。
- main函数中会根据args参数选择对应的实验类，包括DFN(原始的名字) 以及COM对比实验的类。
- 函数需要接受一个**yaml文件**作为参数，yaml文件中包含了实验的超参数以及数据集的路径等信息。
- 根据args执行 experiment_DFN

THU.yaml
- 在本文中只用到expert_list 也就是morlet_wave 的专家知识构建的算子，层数为3 scale为4
- 特征采用 文中的entropy kurtosis mean的特征
- 本文没有采用logic 算子，因此在推理层只用恒等变换代替，等价于通道的全连接层
- 本文没有考虑剪枝的问题，是future work
- save_dir 放到了 code文件夹之外，可能会存在路径问题，需要注意


experiment_DFN
- DFN 实验类继承了fine-tune 的类 ，fine-tune的类继承了base的类 这个是历史遗留问题。
- 实验除了pytorch基本的分类任务之外还实现了
- 1. 保存模型
- 2. 保存模型的参数
- 3. 根据 args mixup的数据增强
- 4. 早停 (少样本其实不需要) 早停会load 模型，但是不是最好的模型，是最后一个模型，所有存在问题
- 5. 梯度裁剪
- 6. metric_recoder
- 7. 可视化不同epoch的权重
- 8. 整理成metric 的字典并且log 进wandb中
- 9. 保存equation 的方法， 这个是历史遗留问题。
- 10. 这个字符串还是挺帅的![alt text](image-1.png)
- 11. 剪枝算法，其实应该现有基础模型，再独立剪枝的类比较方便，所以这一版可以废弃。
- 12. generalization 的类 继承与DFN ，但是单个数据集也够了。

symbolic_layer
..

analysis.ipynb
- 用于分析实验结果的notebook
- 没有体现在论文中的实验
- 1. 变分辨率，降采样数据去验证模型的泛化能力，效果不好
- 2. 权重，阐述意义不明显
- 3. 对比方法是用 10Hz 训练的记得在配置文件中改成10hz
- 4. 泛化任务则改成1Hz ，绘制图片单独保存。


