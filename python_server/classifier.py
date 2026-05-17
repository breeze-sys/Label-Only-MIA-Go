from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


mycfg = {
    'CNN3':  [64, 'M', 64, 'M', 128, 'M', 512, 'M'],
    'CNN4':  [64, 'M', 128, 'M', 128, 'M', 512, 'M'],
    'CNN5':  [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'CNN6':  [64, 'M', 128, 'M', 512, 'M', 512, 'M'],
    'CNN7':  [64, 'M', 256, 'M', 512, 'M', 512, 'M'], ### targetmodel
    'CNN8':  [64, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN9':  [64, 64, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN10':  [64, 128, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN11':  [64, 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M'],

}
parameters = {
    'CIFAR10':  [2, 10],
    'CIFAR100': [2, 100],
    'GTSRB':    [3, 43],
    'Face':     [5, 19],
    'TinyImageNet': [3, 200],
}
class CNN(nn.Module):
    def __init__(self, CNN_name, dataset, dropout=False, dropout_p=0.5):
        super(CNN, self).__init__()
        self.dataset = dataset
        self.query_num = 0
        self.features = self._make_layers(mycfg[CNN_name])
        if dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, parameters[self.dataset][1]) )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, parameters[self.dataset][1]) )
        
    def forward(self, x):
        self.query_num += 1
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x, track_running_stats=True),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=parameters[self.dataset][0], stride=parameters[self.dataset][0])]
        return nn.Sequential(*layers)

class MemGuard(nn.Module):
    def __init__(self):
        super(MemGuard, self).__init__()

    def forward(self, logits):
        scores = F.softmax(logits, dim=1)#.cpu().numpy()
        n_classes = scores.shape[1]
        epsilon = 1e-3
        on_score = (1. / n_classes) + epsilon
        off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
        predicted_labels = scores.max(1)[1]
        defended_scores = torch.ones_like(scores) * off_score
        defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
        return defended_scores
     
