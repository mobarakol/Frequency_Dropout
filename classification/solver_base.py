import abc
import os
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import *
from resnet import *
from resnext import *
from vgg import VGG16_conv
from data import get_data
import wide_resnet


class BaseSolver(metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.train_data, self.test_data, self.args = get_data(args)
        self.cuda = torch.cuda.is_available()

        if self.args.alg == 'normal':
            self.model = CNNNormal(
                    nc=self.args.in_dim,
                    num_classes=self.args.num_classes,
            )
        elif self.args.alg == 'vgg':
            args.lr = 1e-2
            self.model = VGG16_conv(
                    self.args.num_classes,
                    args=args,
            )
        elif self.args.alg == 'res':
            self.model = ResNet18(self.args)
        elif self.args.alg == 'resnext':
            #self.args.batch_size = 128
            self.model = resnext50(self.args)
        elif self.args.alg == 'wrn':
            self.model = wide_resnet.Wide_ResNet(52, 2, 0.3, self.args.num_classes, args)

        self.optim = optim.SGD(
                self.model.parameters(), 
                lr=args.lr,
                weight_decay=5e-4, 
                momentum=0.9,
        )

        if self.cuda:
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()

    def test(self):
        self.model.eval()
        total, correct = 0, 0
        loss_per_iter_test = []
        for images, labels in self.test_data:
            if self.cuda:
                images = images.cuda()

            with torch.no_grad():
                preds = self.model(images)
                loss = self.ce_loss(preds, labels.cuda())
                loss_per_iter_test.append(loss.item())
                preds = torch.argmax(preds, dim=1).cpu().numpy()
                correct += accuracy_score(labels, preds, normalize=False)
                total += images.size(0)

        self.model.train()
        return correct / total * 100, np.mean(loss_per_iter_test)


    def save_model(self):
        weight_dir = './weights' + str(self.args.seed)
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        if self.args.dropout_p_all == [1,1,1]:
            filename = os.path.join(weight_dir,  'model_{}_data_{}_base.tar'.format(self.args.alg, self.args.dataset))
        elif self.args.use_cbs:
            filename = os.path.join(weight_dir,  'model_{}_data_{}_cbs.tar'.format(self.args.alg, self.args.dataset))
        elif self.args.use_gf:
            filename = os.path.join(weight_dir,  'model_{}_data_{}_gf.tar'.format(self.args.alg, self.args.dataset))
        else:
            filename = os.path.join(weight_dir,  'model_{}_data_{}_fd.tar'.format(self.args.alg, self.args.dataset))
        torch.save(self.best_model.state_dict(), filename)


    @abc.abstractmethod
    def solve(self):
        pass
