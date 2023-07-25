# Beyond Individual Input for Deep Anomaly Detection on Tabular Data

[![arXiv](https://img.shields.io/badge/arXiv-2305.15121-b31b1b.svg)](https://arxiv.org/abs/2305.15121)

## Overview

This repo contains the code to run the experiments presented in our paper "Beyond Individual Input for Deep Anomaly Detection on Tabular Data".

The ``npt`` folder contains the main code to run the experiments. This folder was originally forked from https://github.com/OATML/non-parametric-transformers and was adapted to our problem, please cite their work if you use this code.

## Abstract

Anomaly detection is crucial in various domains, such as finance, healthcare, and cybersecurity. In this paper, we propose a novel deep anomaly detection method for tabular data that leverages Non-Parametric Transformers (NPTs), a model initially proposed for supervised tasks, to capture both feature-feature and sample-sample dependencies. In a reconstruction-based framework, we train the NPT model to reconstruct masked features of normal samples. We use the model's ability to reconstruct the masked features during inference to generate an anomaly score. To the best of our knowledge, our proposed method is the first to combine both feature-feature and sample-sample dependencies for anomaly detection on tabular datasets. We evaluate our method on an extensive benchmark of tabular datasets and demonstrate that our approach outperforms existing state-of-the-art methods based on both the F1-Score and AUROC. Moreover, our work opens up new research directions for exploring the potential of NPTs for other tasks on tabular data. 

## Installation

Set up and activate the Python environment by executing

```
conda env create -f environment.yml
```

## Datasets

To download all datasets at once, with `wget`:
```
bash get_dataset_wget.sh
```
with `curl`:
```
bash get_dataset_curl.sh
```

The `data` folder contains the synthetic data, `separable`, used to run the experiments in section 5.1 of the paper.

To add a custom dataset:
- Construct a dataset class inheriting the `BaseDataset` class from `npt/datasets/base.py`.
- Add this dataset class to the imports and in the `DATASET_NAME_TO_DATASET_MAP` dictionnary in the `npt/column_encoding_dataset.py` file.

## Experiments

To run the experiments for each dataset:
```
source ./scripts/abalone.sh
```
where ``abalone`` can be replaced by any dataset in the paper. By default we set either 8 or 4 as the number of GPUs. To change the number of GPUs:
modify the ``.sh`` files accordingly with 
```
--nnodes=$NUMBER_OF_NODE --nproc_per_node=$NUMBER_OF_GPUS_PER_NODE
``` 
```
--mp_nodes $NUMBER_OF_NODE        #number of computing nodes
--mp_gpus $TOTAL_NUMBER_OF_GPUS   #total number of gpus
``` 

For mono-GPU or CPU only:
```
source ./scripts/cpu/abalone.sh
```

### Conda bugs

You may face problems related to conda when launching the `.sh` files. A workaround is to add the following in the file (before activating the environment),
```
source ~/anaconda3/etc/profile.d/conda.sh
```
where `~/` can be replaced accordingly depending on the location of your anaconda installation. 

## Citation

If you use this code for your work, please cite our paper
[Paper](https://arxiv.org/abs/2305.15121) as

```bibtex
@misc{thimonier2023individual,
      title={Beyond Individual Input for Deep Anomaly Detection on Tabular Data}, 
      author={Hugo Thimonier and Fabrice Popineau and Arpad Rimmel and Bich-Liên Doan},
      year={2023},
      eprint={2305.15121},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```