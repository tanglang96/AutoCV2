import logging
import sys

import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

import src

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class ResNet18(models.ResNet):
    Block = BasicBlock

    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = BasicBlock
        super(ResNet18, self).__init__(Block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)  # resnet18
        self.norm = None
        if in_channels == 3:
            self.stem = torch.nn.Sequential(
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                src.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
            )

        self.fc = torch.nn.Linear(512 * Block.expansion, num_classes, bias=False)
        self._half = False
        self._class_normalize = True
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init(self, model_dir, gain=1.):
        sd = model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir)
        del sd['fc.weight']
        del sd['fc.bias']
        self.load_state_dict(sd, strict=False)

        for idx in range(len(self.stem)):
            m = self.stem[idx]
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                LOGGER.debug('initialize stem weight')
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')
        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        LOGGER.debug('initialize layer4')

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        inputs = self.norm(inputs)
        inputs = self.stem(inputs)
        logits = models.ResNet.forward(self, inputs)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, src.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss

        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

    def half(self):
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, torch.nn.BatchNorm2d):
                module.half()
            else:
                module.float()
        self._half = True
        return self
