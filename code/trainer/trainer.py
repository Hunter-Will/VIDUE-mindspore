import os
import mindspore as ms
import mindspore.nn as nn
from trainer.myscheduler import MyScheduler


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.device = 'cpu' if self.args.cpu else 'cuda'+':'+str(args.n_GPUs-1)
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp

        if args.load != '.':
            ms.load_param_into_net(self.optimizer,ms.load_checkpoint(os.path.join(ckp.dir, 'optimizer.ckpt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step(ckp.loss_log[-1])

    def make_optimizer(self):
        kwargs = {'learning_rate': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.opti == 'Adam':
            return nn.Adam(self.model.get_parameters(), **kwargs)
        else:
            return nn.AdaMax(self.model.get_parameters(), **kwargs)  #ax
    
    def make_scheduler(self):
        #return lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.args.patience, verbose=True)
        return MyScheduler(self.optimizer, mode='min', factor=0.5, patience=self.args.patience, verbose=True)
    
    def train(self):
        pass

    def test(self):
        pass

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
