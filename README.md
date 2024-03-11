# Robust-Dataset-Distillation

This is an official PyTorch implementation of ["Group Distributionally Robust Dataset Distillation with Risk Minimization"](https://arxiv.org/pdf/2402.04676.pdf).

## Abstract

Dataset distillation (DD) has emerged as a widely adopted technique for crafting a synthetic dataset that captures the essential information of a training dataset, facilitating the training of accurate neural models. Its applications span various domains, including transfer learning, federated learning, and neural architecture search. The most popular methods for constructing the synthetic data rely on matching the convergence properties of training the model with the synthetic dataset and the training dataset. However, targeting the training dataset must be thought of as auxiliary in the same sense that the training set is an approximate substitute for the population distribution, and the latter is the data of interest. Yet despite its popularity, an aspect that remains unexplored is the relationship of DD to its generalization, particularly across uncommon subgroups. That is, how can we ensure that a model trained on the synthetic dataset performs well when faced with samples from regions with low population density? Here, the representativeness and coverage of the dataset become salient over the guaranteed training error at inference. Drawing inspiration from distributionally robust optimization, we introduce an algorithm that combines clustering with the minimization of a risk measure on the loss to conduct DD. We provide a theoretical rationale for our approach and demonstrate its effective generalization and robustness across subgroups through numerical experiments.


## Getting Started

Download the repo:
```bash
git clone https://github.com/Mming11/RobustDatasetDistillation.git
```

## Requirements
- The code has been tested on PyTorch 2.0.1.   
- To run the code, install package ```pip install fast-pytorch-kmeans```

## Experiment Commands
### SVHN
```
python distill_rdd.py --dataset=svhn --ipc=10 --layer=2 --eval_it=100 --space wp --learn_g --lr_g=0.01 --eval_mode=SVHN --depth=3
```

### CIFAR10
```
python distill_rdd.py --dataset=CIFAR10 --ipc=10 --layer=2 --eval_it=100 --space wp --eval_mode=CIFAR --depth=3
```

### ImageNet
```
python distill_rdd.py --dataset=imagenet-a --space=wp --eval_it=100 --layer=16 --ipc=10 --data_path={path_to_dataset}
```

## Test Commands
- Put the distilled data at ```./test/results```.
- Set ```--data_dir``` and ```--imagenet_dir``` in ```test/argument.py``` to the folder containing the original dataset.
- We provide some tests of robustness metrics. You need to process the original dataset via ```data_processing.ipynb``` and then place it in the corresponding path. For detailed testing commands, please see [test/README.md].

For example, to evaluate (10 images/class) on CIFAR-10 , run
```
python test_glad.py -d cifar10 -n convnet -s glad --ipc 10
```

You can also test the performance of other datasets by changing the parameters.


## Acknowledgement
This project is mainly developed based on the following works:
- [GLaD](https://github.com/georgecazenavette/glad)
- [IDC](https://github.com/snu-mllab/efficient-dataset-condensation)

## Citation
If you find this work helpful, please cite:
```
@article{vahidian2024group,
  title={Group Distributionally Robust Dataset Distillation with Risk Minimization},
  author={Vahidian, Saeed and Wang, Mingyu and Gu, Jianyang and Kungurtsev, Vyacheslav and Jiang, Wei and Chen, Yiran},
  journal={arXiv preprint arXiv:2402.04676},
  year={2024}
}
```