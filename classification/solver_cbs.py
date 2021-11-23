import torch
import copy
import os
import torch.optim as optim
import torch.utils.data as data
import math

from arguments import get_args

from models import *
from data import get_data
from solver_base import BaseSolver

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class CBSSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

        self.decay_epoch = 50 if self.args.alg == 'vgg' else 30
        self.stop_decay_epoch = self.decay_epoch * 3 + 1

    def solve(self):
        best_epoch, best_acc = 0, 0
            
        self.ce_loss = F.cross_entropy
        loss_per_epoch_train = []
        loss_per_epoch_test = []
        acc_per_epoch_test = []
        for epoch_count in range(self.args.num_epochs):
            if not self.args.use_cbs:
                self.model.module.get_new_kernels()
            else:
                self.model.module.get_new_kernels_cbs(epoch_count)
                  
            self.model = self.model.cuda()

            if epoch_count is not 0 and epoch_count % self.decay_epoch == 0 \
                    and epoch_count < self.stop_decay_epoch:
                print(epoch_count, self.decay_epoch, self.stop_decay_epoch)
                for param in self.optim.param_groups:
                    param['lr'] = param['lr'] / 10 
            loss_per_iter_train =[]
            for images, labels in self.train_data:
                images = images.cuda()
                labels = labels.cuda()
                preds = self.model(images)
                loss = self.ce_loss(preds, labels)
                loss_per_iter_train.append(loss.item()) 
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            loss_per_epoch_train.append(np.mean(loss_per_iter_train))

            if epoch_count % 1 == 0:
                accuracy, loss_test = self.test()
                loss_per_epoch_test.append(loss_test)
                acc_per_epoch_test.append(accuracy)
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_epoch = epoch_count
                    self.best_model = copy.deepcopy(self.model)
                    self.save_model()

                print('current epoch: {}, accuracy: {:.2f}, best epoch: {}, best acc: {:.2f} '.format(
                        epoch_count, accuracy, best_epoch, best_acc))
        if not os.path.exists(self.args.log_path):
            os.mkdir(self.args.log_path)
        if self.args.dropout_p_all == [1,1,1]:
            loos_filename = os.path.join(self.args.log_path, 'model_{}_data_{}_base'.format(self.args.alg, self.args.dataset))
        elif self.args.use_cbs:
            loos_filename = os.path.join(self.args.log_path, 'model_{}_data_{}_cbs'.format(self.args.alg, self.args.dataset))
        elif self.args.use_gf:
            loos_filename = os.path.join(self.args.log_path, 'model_{}_data_{}_gf_{}'.format(self.args.alg, self.args.dataset, self.args.kernel_size))
        else:
            loos_filename = os.path.join(self.args.log_path, 'model_{}_data_{}_fd_{}'.format(self.args.alg, self.args.dataset, self.args.kernel_size))
        np.save('{}_train.npy'.format(loos_filename), loss_per_epoch_train)
        np.save('{}_test.npy'.format(loos_filename), loss_per_epoch_test)
        np.save('{}_acc_test.npy'.format(loos_filename), acc_per_epoch_test)