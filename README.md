# Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective

This is the official PyTorch implementation of the **ReBAT (ReBalanced Adversarial Training)** algorithm proposed in our NeurIPS 2023 paper [Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective](https://arxiv.org/abs/2310.19360), by *Yifei Wang\*, Liangchen Li\*, Jiansheng Yang, Zhouchen Lin and Yisen Wang*.

## Requirements

- Our code is compatiable with PyTorch 2.0.0.
- Please install [AutoAttack](https://arxiv.org/abs/2003.01690) from their official [codebase](https://github.com/fra31/auto-attack).

## Training

First, please run the following command to generate a validation set:

```python
python3 generate_validation.py
```

Train a PreActResNet-18 model on CIFAR-10 with ReBAT:

```python
CUDA_VISIBLE_DEVICES=0 python3 train_cifar_wa.py --val \
	--fname cifar10_res18 \
	--model PreActResNet18 \
	--chkpt-iters 10 \
	--lr-factor 1.5 \
	--beta 1.0
```

Train a PreActResNet-18 model on CIFAR-10 with ReBAT++:

```python
CUDA_VISIBLE_DEVICES=0 python3 train_cifar_wa.py --val \
	--fname cifar10_res18_pp \
	--model PreActResNet18 \
	--chkpt-iters 10 \
	--lr-factor 1.7 \
	--beta 1.0 \
	--stronger-attack \
	--stronger-epsilon 10 \
	--stronger-attack-iters 12
```

Train a WideResNet-34-10 model on CIFAR-10 with ReBAT+CutMix:

```python
CUDA_VISIBLE_DEVICES=0 python3 train_cifar_wa.py --val \
	--fname cifar10_wrn34_cutmix \
	--model WideResNet \
	--chkpt-iters 10 \
	--lr-factor 4.0 \
	--beta 2.0 \
	--cutmix
```

## Evaluation

Evaluate a PreActResNet-18 model on CIFAR-10:

```python
CUDA_VISIBLE_DEVICES=0 python3 train_cifar_wa.py --eval \
	--fname cifar10_res18 \
	--model PreActResNet18 \
	--resume 200
```

Evaluate a WideResNet-34-10 model on CIFAR-10:

```python
CUDA_VISIBLE_DEVICES=0 python3 train_cifar_wa.py --eval \
	--fname cifar10_wrn34_cutmix \
	--model WideResNet \
	--resume 200
```

## Checkpoint Release

We provide both best and last checkpoints under several experiment settings on [Google Drive](https://drive.google.com/drive/folders/1pUIt-C0_bPpBf6QrsR2qfFyUR227UwqW?usp=share_link). Please download and place the subfolders under ./exps/ for evaluation.

## Citation

Please cite our paper if you find our work useful.

```
@article{wang2023balance,
  title={Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective},
  author={Wang, Yifei and Li, Liangchen and Yang, Jiansheng and Lin, Zhouchen and Wang, Yisen},
  journal={Advances in neural information processing systems},
  volume={36},
  year={2023}
}
```

