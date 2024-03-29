# TON_TII
CN version[CN|](README_CN.md)[EN|](README.md)
## Project Overview

TON_TII is a project aimed at constructing a fully interpretable diagnostic model. By integrating deep learning and symbolic regression techniques, it offers a novel approach to understanding and interpreting the decision-making process of models. This method is particularly suited for applications requiring high levels of transparency and interpretability, such as medical diagnostics and fault detection.

## Environment Setup

This project uses `conda` to manage dependencies. Please ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, then run the following command to create and activate an environment containing all dependencies:

```bash
conda create --name ton_tii_env --file requirements.txt
```

Most environments are not mandatory, pytorch

## How to Run

python main.py experiment/THU.yaml


## Program Structure TDOO fix bugs

- data_dir/: Contains data loaders and data files.
- experiment/: Contains experiment configuration files and scripts.
- figs/: Stores generated figures.
- model/: Contains model definition files.
- plot_dir/: Stores generated plots.
- post_analysis/: Contains post-processing and analysis scripts.
- reference/: Contains references or related materials.
- save_dir/: Stores trained models and other persistent results.
- utils/: Contains utilities and helper functions.
- main.py: The main entry script of the project.

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
│   ├── comparison_net.py
│   ├── symbolic_base.py
│   ├── symbolic_layer.py
│   └── test.ipynb
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
