import os
from importlib import import_module
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O
from loss.hard_example_mining import HEM
from loss.losses import SupConLoss
from loss.ordinal_losses import OrdinalSupConLoss

class Loss(nn.LossBase):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')
        self.loss = []
        self.loss_module = nn.CellList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'SupCon':
                loss_function = SupConLoss()
            elif loss_type == 'OrdinalSupCon':
                loss_function = OrdinalSupConLoss()
            elif loss_type == 'CrossEntropy':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'HEM':
                loss_function = HEM()
            else:
                raise NotImplementedError('Loss type [{:s}] is not found'.format(loss_type))

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = O.zeros((1, len(self.loss)))

        self.loss_module

        if args.load != '.':
            self.load(ckp.dir, cpu=args.cpu)

    def construct(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item(0)
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item(0)

        return loss_sum

    def start_log(self):
        pass
        #self.log = O.cat((self.log, O.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1]=self.log[-1].div(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c.numpy() / n_samples))

        return ''.join(log)

    def get_loss_module(self):
        return self.loss_module

    def save(self, apath):
        ms.save_checkpoint(self, os.path.join(apath, 'loss.ckpt'))
        ms.save_checkpoint([{"name":"loss.log","data":self.log}], os.path.join(apath, 'loss_log.ckpt'))

    def load(self, apath):
        ms.load_param_into_net(self, ms.load_checkpoint(os.path.join(apath, 'loss.ckpt')))
        self.log = ms.load_checkpoint(os.path.join(apath, 'loss_log.ckpt'))["loss.log"]