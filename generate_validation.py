import torch
import torchvision
import numpy as np
from utils import cifar

np.random.seed(0)
m = 50000
P = np.random.permutation(m)

n = 1000


# CIFAR-10
dataset = cifar('../cifar-data')

val_data = dataset['train']['data'][P[:n]]
val_labels = [dataset['train']['labels'][p] for p in P[:n]]
train_data = dataset['train']['data'][P[n:]]
train_labels = [dataset['train']['labels'][p] for p in P[n:]]

dataset['train']['data'] = train_data
dataset['train']['labels'] = train_labels
dataset['val'] = {
    'data': val_data,
    'labels': val_labels
}
dataset['split'] = n
dataset['permutation'] = P

torch.save(dataset, 'cifar10_validation_split.pth')

# CIFAR-100
dataset = cifar('../cifar-data', num_classes=100)

val_data = dataset['train']['data'][P[:n]]
val_labels = [dataset['train']['labels'][p] for p in P[:n]]
train_data = dataset['train']['data'][P[n:]]
train_labels = [dataset['train']['labels'][p] for p in P[n:]]

dataset['train']['data'] = train_data
dataset['train']['labels'] = train_labels
dataset['val'] = {
    'data': val_data,
    'labels': val_labels
}
dataset['split'] = n
dataset['permutation'] = P

torch.save(dataset, 'cifar100_validation_split.pth')
