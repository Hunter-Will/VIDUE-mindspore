import decimal
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O

class NetwithLoss(nn.Cell):
    def __init__(self,net,loss_fn):
        super(NetwithLoss,self).__init__()
        self.net=net
        self.loss_fn=loss_fn
        
    def construct(self,inputs, label_list):
        _, out=self.net(inputs)
        bsz = label_list.shape[0]
        f1, f2 = O.split(out, [bsz, bsz], 0)
        features = O.cat([f1.unsqueeze(1), f2.unsqueeze(1)], 1)

        loss = self.loss_fn(features, label_list)
        return loss
        

class Trainer_Prior_predict(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Prior_predict, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer-Prior_predict")
        self.m = args.m
        self.n = args.n
        self.min = 0

    def make_optimizer(self):
        kwargs = {'learning_rate': self.args.lr, 'weight_decay': self.args.weight_decay}

        return nn.Adam(self.model.get_model().get_parameters(),**kwargs)



    def train(self):
        print("Now training")
        epoch = self.scheduler.last_epoch #+ 1
        lr = self.optimizer.parameters_dict()['learning_rate'].value().numpy()   #self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(float(lr))))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.

        train_net=nn.TrainOneStepCell(NetwithLoss(self.model.get_model(),self.loss), self.optimizer)
        
        for batch, (input_list, label_list, _, input_filenames, exp) in enumerate(self.loader_train):
            b,n,c,h,w = input_list.shape
            images = input_list.reshape([b*n,c,h,w])

            input = images.float()
            label_list=label_list.float()
            loss=train_net(input,label_list)

            self.ckp.report_log(loss.item(0))

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[mid: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1].numpy() / (batch + 1),
                    self.loss.display_loss(batch),
                    mid_loss_sum / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train.dataset))
        self.min = self.ckp.loss_log[-1]
        self.scheduler.step(self.ckp.loss_log[-1])

    def test(self):
        epoch = self.scheduler.last_epoch #+ 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        self.model.get_model().set_train(False)
        total_num = 0.
        total_deblur_PSNR =0.
        tqdm_test = tqdm(self.loader_test, ncols=80)
        for idx_img, (input_list, label_list, _, input_filenames, exp) in enumerate(tqdm_test):

            b,n,c,h,w = input_list.shape
            images = input_list.reshape([b*n,c,h,w])
            input = images.float()
            label_list=label_list.float()

            _, out = self.model(input)

            bsz = label_list.shape[0]
            f1, f2 = O.split(out, [bsz, bsz], 0)
            features = O.cat([f1.unsqueeze(1), f2.unsqueeze(1)], 1)

            loss = self.loss(features, label_list)
            self.ckp.report_log(self.min, train=False)

        self.ckp.end_log(len(self.loader_test.dataset), train=False)
        best = self.ckp.psnr_log.min(0,return_indices=True)
        self.ckp.write_log('[{}]\taverage SupCon: {:.5f} (Best: {:.5f} @epoch {})'.format(
            self.args.data_test,
            self.ckp.psnr_log[-1].numpy(),
            best[0].numpy(), best[1] + 1))  #
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))  #


