import decimal
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O

class NetwithLoss(nn.Cell):
    def __init__(self,net,loss_fn,mid_loss_fn,mid_loss_weight):
        super(NetwithLoss,self).__init__()
        self.net=net
        self.loss_fn=loss_fn
        self.mid_loss_fn=mid_loss_fn
        self.mid_loss_weight=mid_loss_weight
        
    def construct(self, inputs, gt):
        out, flows = self.net(inputs)  #, epoch
        output = O.cat(out, 1)
        loss = self.loss_fn(output, gt)  # + loss_reblur + loss_pair/4
        b, c, h, w = flows.shape
        flows = flows.view(b, -1, 2, h, w)
        flows = flows.view(-1, 2, h, w)
        mid_loss = self.mid_loss_fn(flows)
        loss = loss + self.mid_loss_weight * mid_loss #+ mid_loss_warp
        return O.stack((loss,mid_loss))

class Trainer_UNet(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_UNet, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer-VIDUE")
        self.m = args.m
        self.n = args.n
        self.blur_deg = args.blur_deg
        self.args = args

    def make_optimizer(self):
        kwargs = {'learning_rate': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.opti == 'Adam':
            print("Using Adam...")
            return nn.Adam([{"params": self.model.get_model().UNet.get_parameters()},
                                 {"params": self.model.get_model().refine.get_parameters()}], **kwargs)
        else:
            print("Using Adamax...")
            return nn.AdaMax([{"params": self.model.get_model().UNet.get_parameters()},
            {"params": self.model.get_model().refine.get_parameters()},
            {"params": self.model.get_model().motion.get_parameters()}],**kwargs)  #, "lr": 1e-6

    def charbonnier(self, x, alpha=0.25, epsilon=1.e-9):
        return O.pow(O.pow(x, 2) + epsilon ** 2, alpha)

    def smoothness_loss(self, flow):
        b, c, h, w = flow.shape
        v_translated = O.cat((flow[:, :, 1:, :], O.zeros((b, c, 1, w))), -2)
        h_translated = O.cat((flow[:, :, :, 1:], O.zeros((b, c, h, 1))), -1)
        s_loss = self.charbonnier(flow - v_translated) + self.charbonnier(flow - h_translated)
        s_loss = O.sum(s_loss, 1) / 2

        return O.sum(s_loss) / b

    def train(self):
        print("Now training")
        epoch = self.scheduler.last_epoch #+ 1
        lr = self.optimizer.parameters_dict()['learning_rate'].value().numpy()   #self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(float(lr))))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.
        mid_loss_sum_warp = 0.

        train_net=nn.TrainOneStepCell(NetwithLoss(self.model.get_model(),self.loss,
                                                  self.smoothness_loss,self.args.mid_loss_weight),
                                      self.optimizer)
        
        for batch, (input_list, gt_list, output_filenames, input_filenames, exp) in enumerate(self.loader_train):
            inputs=[input_list[:,i,:,:,:] for i in range(input_list.shape[1])]
            inputs = [inp.float() for inp in inputs]
            b,n,c,h,w=gt_list.shape
            gt = gt_list.reshape([b,n*c,h,w]).float()
            
            loss_t=train_net(inputs,gt)
            loss=loss_t[0]
            mid_loss=loss_t[1]
            
            mid_loss_sum = mid_loss_sum + mid_loss.item(0)
            self.ckp.report_log(loss.item(0))

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[mid: {:.4f}][mid_warp: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1].numpy() / (batch + 1),
                    self.loss.display_loss(batch),
                    mid_loss_sum.numpy() / (batch + 1),
                    mid_loss_sum_warp / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train.dataset))
        self.scheduler.step(self.ckp.loss_log[-1])

    def test(self):
        epoch = self.scheduler.last_epoch #+ 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.start_log(train=False)
        self.model.get_model().set_train(False)
        total_num = 0.
        total_deblur_PSNR =0.
        tqdm_test = tqdm(self.loader_test, ncols=80)
        for idx_img, (input_list, gt_list, output_filenames, input_filenames, exp) in enumerate(tqdm_test):
            exp = exp//self.blur_deg
            inputs=[input_list[:,i,:,:,:] for i in range(input_list.shape[1])]
            inputs = [inp.float() for inp in inputs]
            b,n,c,h,w=gt_list.shape
            gt = gt_list.reshape([b,n*c,h,w]).float()
            gt = gt_list.float()
            
            out, _ = self.model(input)
            PSNR = 0.
            for i in range(len(out)):
                PSNR_item = utils.calc_psnr(gt[i], out[i], rgb_range=self.args.rgb_range)
                if i == exp//2:
                    deblur_PSNR = PSNR_item
                PSNR += PSNR_item / len(out)
            total_deblur_PSNR += deblur_PSNR
            total_num += 1
            self.ckp.report_log(PSNR, train=False)

            if self.args.save_images:
                save_list = out
                save_list.append(gt[exp//2])
                save_list.append(input[self.args.n_sequence//2-1])
                save_list = utils.postprocess(save_list,rgb_range=self.args.rgb_range,
                                            ycbcr_flag=False, device=self.device)
                self.ckp.save_images(output_filenames, save_list, epoch, exp)

        self.ckp.end_log(len(self.loader_test.dataset), train=False)
        best = self.ckp.psnr_log.max(0,return_indices=True)
        self.ckp.write_log('[{}]\taverage Deblur_PSNR: {:.3f} Total_PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
            self.args.data_test,
            total_deblur_PSNR / total_num,
            self.ckp.psnr_log[-1].numpy(),
            best[0].numpy(), best[1] + 1))  #
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))  #


