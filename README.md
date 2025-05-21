This is the official repository for the paper [Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method](https://arxiv.org/abs/2409.14781) by *Weichao Zhang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng* <br>


## Overview

we proposes DC-PDD to improve methods that directly rely on token probabilities for pretraining data detection, which tend to misclassify non-training texts containing many common words as training texts. The key idea of DC-PDD is to calibrate the token probabilities and thereby make them more informative signals for detection. The calibration process is achieved by computing the cross-entropy (i.e., the divergence) between the token probability distribution and the token frequency distribution. To facilitate the evaluation of pretraining data detection for LLMs, we introduce a new benchmark named **PatentMIA**, specifically designed for Chinese-language pretraining data detection.

![GitHub Logo](figures/DC-PDD.png)

## PatentMIA
The PatentMIA dataset serves as a benchmark designed to evaluate pretraining data detection methods, specifically in detecting Chinese-language pretraining data from models that are open-source Chinese LLMs released between January 1, 2023 and March 1, 2024 (e.g., Qwen1.5)

The dataset contains non-training and training data:
- non-training data includes text snippets crawled from Google Patent websites published after March 1, 2024.
- training data includes text snippets crawled from Google Patent websites published before January 1, 2023.

## DC-PDD (& baselines)

We first obtain the token probability distribution by querying the LLM with the text.
```
python src/com_pro_dis.py --tar_mod <model_name> --ref_dat <data_file> --data <data_file>
```

Next, we use a large-scale publicly available corpus as a reference corpus to obtain an estimation of the token frequency distribution since an LLM’s pretraining corpus is not accessible usually. 
```
python src/com_fre_dis.py --model <model_name> --ref_dat <reference_dataset>
```

We then calibrate the token probabilities by comparing the token probability distribution to the token frequency distribution to calibrate the token probability for each token, and derive a score for pretraining data detection based on the calibrated token probabilities.
```
python src/com_det_sco.py --tar_mod <model_name> --data <data_file> --max_cha <text-length> --lang <language> --a <hyperparameter of DC-PDD>
```

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{zhang2024pretraining,
  title={Pretraining data detection for large language models: A divergence-based calibration method},
  author={Zhang, Weichao and Zhang, Ruqing and Guo, Jiafeng and de Rijke, Maarten and Fan, Yixing and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2409.14781},
  year={2024}
}
```
