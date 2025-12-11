# CSOG: Cross-disease Knowledge Transfer Framework for Small-sample Omics Analysis with GNNs

[![Paper](https://img.shields.io/badge/Paper-BIBM25-blue)](https://arxiv.org/abs/xxx.xxxx) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)

Official implementation of the paper **"A Cross-disease Knowledge Transfer Framework for Small-sample Omics Analysis with GNNs"** (accepted by BIBM 2025).

## Citation
If you use CSOG in your research, please cite our paper:
```bibtex
@inproceedings{chaosong2025csog,
  title={A Cross-disease Knowledge Transfer Framework for Small-sample Omics Analysis with GNNs},
  author={Chao Song, Kunyang Xian, Peng Yao, Lu Gan, Ruilin Hu, Li Lu, and Yu Cao},
  booktitle={Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2025},
  organization={IEEE}
}
```
## Overview
CSOG (Cross-disease knowledge transfer framework for Small-sample Omics analysis with GNNs) addresses the challenge of overfitting and poor generalization in small-sample disease omics analysis. It leverages a **pretraining-freezing fine-tuning paradigm** to transfer biological network knowledge from large-sample multi-omics data to small-sample single-omics target diseases, achieving state-of-the-art classification performance.

### Key Features
- First dedicated cross-disease knowledge transfer framework for small-sample GNN-based omics analysis.
- Integrates multi-omics representation optimization (contrastive learning + graph regularization) and minimal fine-tuning.
- Reduces trainable parameters to avoid overfitting, while preserving learned biological network patterns via parameter freezing.
- Outperforms 10 state-of-the-art baselines on 3 small-sample omics datasets (ACC, MESO, UCS).

<img width="661" height="480" alt="image" src="https://github.com/user-attachments/assets/4b7e8125-47dc-47ed-975a-0747941c9347" />



## Datasets
We use 4 real-world omics datasets from TCGA (https://gdac.broadinstitute.org/):

| Dataset | Type       | Samples | Omics Types       | Labels                                  |
|---------|------------|---------|-------------------|-----------------------------------------|
| BRCA    | Source     | 875     | mRNA, methylation, miRNA | 5 cancer subtypes (Normal-like, Basal-like, etc.) |
| ACC     | Target     | 77      | mRNA              | 4 pathologic stages (I-IV)              |
| MESO    | Target     | 83      | mRNA              | 4 pathologic N categories (n0-n3)       |
| UCS     | Target     | 56      | mRNA              | 4 pathologic stages (I-IV)              |



## Prerequisites
- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric (PyG) 2.0+

## Running
Run the main.py

