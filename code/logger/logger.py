import mindspore as ms
import mindspore.ops as O
import imageio
import numpy as np
import os
import datetime
import skimage.color as sc

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, args):
        self.args = args
        self.psnr_log = ms.Tensor([0.0])
        self.loss_log = ms.Tensor([0.0])
        self.log_init = False

        if args.load == '.':
            if args.save == '.':
                args.save = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
            self.dir = args.experiment_dir + args.save
        else:
            self.dir = args.experiment_dir + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.loss_log = ms.load_checkpoint(self.dir + '/loss_log.ckpt')["loss_log"][:, -1]
                self.psnr_log = ms.load_checkpoint(self.dir + '/psnr_log.ckpt')["psnr_log"]
                self.log_init = True
                print('Continue from epoch {}...'.format(len(self.psnr_log)))

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            if not os.path.exists(self.dir + '/model'):
                os.makedirs(self.dir + '/model')
        if not os.path.exists(self.dir + '/result/' + self.args.data_test):
            print("Creating dir for saving images...", self.dir + '/result/' + self.args.data_test)
            os.makedirs(self.dir + '/result/' + self.args.data_test)

        print('Save Path : {}'.format(self.dir))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

    def save(self, trainer, epoch, is_best):
        trainer.model.save(self.dir, epoch, is_best)
        ms.save_checkpoint([{"name":"psnr_log","data":self.psnr_log}], os.path.join(self.dir, 'psnr_log.ckpt'))
        ms.save_checkpoint(trainer.optimizer, os.path.join(self.dir, 'optimizer.ckpt'))
        trainer.loss.save(self.dir)
        #self.plot_psnr_log(epoch)

    def save_images(self, filename, save_list, epoch, exposure):
        if self.args.task == 'VideoDeblur':
            length = len(filename)
            postfix = []
            f = filename[0][0].split('.')
            # print(length)
            # print(f)
            dirname = '{}/result/{}/{}'.format(self.dir, self.args.data_test, f[0])
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            for i in range(length):
                f = filename[i][0].split('.')
                curfilename = '{}/{}'.format(dirname, f[1])
                # print(curfilename)
                postfix.append(curfilename)
            postfix.append('{}/{}_{}'.format(dirname, filename[exposure//2][0].split('.')[1], 'gt'))
            postfix.append('{}/{}_{}'.format(dirname, filename[exposure// 2][0].split('.')[1], 'blur'))
        else:
            raise NotImplementedError('Task [{:s}] is not found'.format(self.args.task))
        # print(len(save_list), len(postfix))
        for img, post in zip(save_list, postfix):
            # print(img.size())
            img = img[0].data
            img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            if img.shape[2] == 1:
                img = img.squeeze(axis=2)
            elif img.shape[2] == 3 and self.args.n_colors == 1:
                img = sc.ycbcr2rgb(img.astype('float')).clip(0, 1)
                img = (255 * img).round().astype('uint8')
            imageio.imwrite('{}.png'.format(post), img)

    def start_log(self, train=True):
        if train:
            self.loss_log = O.cat((self.loss_log, O.zeros(1))) if self.log_init else O.zeros(1)
        else:
            self.psnr_log = O.cat((self.psnr_log, O.zeros(1))) if self.log_init else O.zeros(1)

    def report_log(self, item, train=True):
        if train:
            self.loss_log[-1] += item
        else:
            self.psnr_log[-1] += item

    def end_log(self, n_div, train=True):
        if train:
            self.loss_log[-1]=self.loss_log[-1].div(n_div)
        else:
            self.psnr_log[-1]=self.psnr_log[-1].div(n_div)

    def plot_loss_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('Loss Graph')
        plt.plot(axis, self.loss_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'loss.pdf'))
        plt.close(fig)

    def plot_psnr_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')
        plt.plot(axis, self.psnr_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'psnr.pdf'))
        plt.close(fig)

    def done(self):
        self.log_file.close()
