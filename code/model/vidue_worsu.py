import math
import numpy as np
import importlib

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O
import model.blocks as blocks
from model.blocks import Sty_layer
from model.time_prior_predict_weighted import TPP
from model.ETR import ETR_motion_V2
from model.refine.UNet2 import UNet2_V2 as Refine_V
from model.convtranspose2d import ConvTranspose2d

def make_model(args):
    return ResNet(n_inputs=args.n_sequence, n_outputs_e=args.m, n_outputs_r=args.n, blocks=args.n_resblock, feats=args.n_feat, offset_network_path=args.offset_network_path, extractor_path=args.extractor_path, halve=args.halve)


class SynBlock(nn.Cell):
    def __init__(self, nf, ks, exp=5, ro=3):
        super(SynBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf//4, 1, 1, "pad", 0, has_bias=True)
        self.conv2 = nn.Conv2d(1 + nf // 4, exp + ro, 3, 1, "pad", 1, has_bias=True)
        self.se1 = Sty_layer(256, (exp + ro)*2, 4)

        ref1 = [nn.Conv2d(2, 2, kernel_size=ks, stride=1, padding=ks // 2, pad_mode="pad", has_bias=True)
                for _ in range(exp + ro)]
        self.ref1 = nn.SequentialCell(ref1)
        syn1 = [nn.Conv2d(nf//4, nf//4, kernel_size=ks, stride=1, padding=ks // 2, pad_mode="pad", has_bias=True)
                for _ in range(exp+ro)]
        self.syn1 = nn.CellList(syn1)
        syn2 = [blocks.ResBlock(nf//4, nf//4, kernel_size=ks, stride=1)
                          for _ in range(exp+ro)]
        self.syn2 = nn.CellList(syn2)
        syn3 = [blocks.ResBlock(nf // 4, nf // 4, kernel_size=ks, stride=1)
                for _ in range(exp + ro)]
        self.syn3 = nn.CellList(syn3)
        out = [nn.Conv2d(nf//4, 3, kernel_size=ks, stride=1, padding=ks // 2, pad_mode="pad", has_bias=True)
                for _ in range(exp+ro)]
        self.out = nn.CellList(out)
        self.xN = exp+ro

    def warp(self, x, flo):
        
        B, C, H, W = x.shape
        xx = O.arange(0, W).view(1, -1).tile((H, 1))
        yy = O.arange(0, H).view(-1, 1).tile((1, W))
        xx = xx.view(1, 1, H, W).tile((B, 1, 1, 1))
        yy = yy.view(1, 1, H, W).tile((B, 1, 1, 1))
        grid = O.cat((xx, yy), 1).float()
        vgrid = grid + flo # torch.autograd.Variable(grid) + flo   have no idea why this grid should be variable
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].copy() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].copy() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = O.grid_sample(x, vgrid, padding_mode='border')
        mask = O.ones(x.shape) # torch.autograd.Variable(torch.ones(x.size())).cuda()   have no idea why this mask should be variable
        mask = O.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output #, mask

    def construct(self, fea, flows, mmap, vec, out_size):  #
        H, W = out_size
        B, C, _, _ = flows.shape
        in_fea = self.conv1(fea)
        results = []
        flows = O.interpolate(flows, size=out_size, mode='bilinear')
        ref_flows = self.se1(flows, vec)
        flows = flows.view(B, C // 2, 2, H, W)
        ref_flows = ref_flows.view(B, C // 2, 2, H, W)
        mmap = O.interpolate(mmap, size=out_size, mode='bilinear')
        mmap = self.conv2(O.concat([mmap, in_fea],1)) + mmap
        mmap = O.Sigmoid()(mmap)

        f_flows = [ref_flows[:,j]*mmap[:,j:j+1]+flows[:,j]*(1-mmap[:,j:j+1]) for j in range(self.xN)]

        for i in range(self.xN):
            f_flows[i] = self.ref1(f_flows[i]) + f_flows[i]
            fea2 = in_fea*(1-mmap[:,i:i+1])+self.warp(in_fea, f_flows[i])*mmap[:,i:i+1]
            fea3 = self.syn1[i](fea2) + fea2
            fea4 = self.syn2[i](fea3) + fea3
            fea5 = self.syn3[i](fea4) + fea4
            out = self.out[i](fea5)
            results.append(out)

        return results

class UNet(nn.Cell):

    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=[3,3,9,3], n_feat=32, kernel_size=3,n_outputs_e=5, n_outputs_r=3):
        super(UNet, self).__init__()

        InBlock = []

        InBlock.extend([nn.SequentialCell([
            nn.Conv2d(in_channels * n_sequence, n_feat, kernel_size=7, stride=1,
                      padding=7 // 2, pad_mode="pad", has_bias=True),
            nn.LeakyReLU(0.1)]
        )])

        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1, se=False)
                        for _ in range(n_resblock[0])])

        # encoder1
        Encoder_first = [nn.SequentialCell([
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, pad_mode="pad", has_bias=True),
            nn.LeakyReLU(0.01)]
        )]
        Encoder_first.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1, se=False)
                              for _ in range(n_resblock[1])])
        # encoder2
        Encoder_second = [nn.SequentialCell([
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, pad_mode="pad", has_bias=True),
            nn.LeakyReLU(0.01)]
        )]
        Encoder_second.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, se=False)
                               for _ in range(n_resblock[2])])

        # encoder3
        Encoder_third = [nn.SequentialCell([
            nn.Conv2d(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, pad_mode="pad", has_bias=True),
            nn.LeakyReLU(0.01)]
        )]
        Encoder_third.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, se=False)
                               for _ in range(n_resblock[3])])

        # decoder3
        Decoder_third = [nn.SequentialCell([
            # convtranspose2d implemented by myself because stupid ms devs can't code it right even if they have a very right op kernel.
            ConvTranspose2d(n_feat *4, n_feat * 4, kernel_size=3, stride=2, padding=1, output_padding=1, has_bias=True), 
            nn.LeakyReLU(0.01)]
        )]
        Decoder_third.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, se=False)
                          for _ in range(n_resblock[3])])

        # decoder2
        Decoder_second = [nn.SequentialCell([
            ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1, has_bias=True),
            nn.LeakyReLU(0.01)]
        )]
        Decoder_second.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1, se=False)
                          for _ in range(n_resblock[2])])
        # decoder1
        Decoder_first = [nn.SequentialCell([
            ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, has_bias=True),
            nn.LeakyReLU(0.01)]
        )]
        Decoder_first.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1, se=False)
                         for _ in range(n_resblock[1])])

        OutBlock = [blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                    for _ in range(n_resblock[0])]
        OutBlock.append(nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, pad_mode="pad", has_bias=True))

        self.predict_ll = SynBlock(n_feat * 4, ks=3, exp=n_outputs_e, ro=n_outputs_r)
        self.predict_l = SynBlock(n_feat * 2, ks=3, exp=n_outputs_e, ro=n_outputs_r)
        self.predict = SynBlock(n_feat, ks=3, exp=n_outputs_e, ro=n_outputs_r)

        self.inBlock = nn.SequentialCell(InBlock)
        self.encoder_first = nn.SequentialCell(Encoder_first)
        self.encoder_second = nn.SequentialCell(Encoder_second)
        self.encoder_third = nn.SequentialCell(Encoder_third)
        self.decoder_third = nn.SequentialCell(Decoder_third)
        self.decoder_second = nn.SequentialCell(Decoder_second)
        self.decoder_first = nn.SequentialCell(Decoder_first)
        self.outBlock = nn.SequentialCell(OutBlock)

        self.se3 = Sty_layer(n_feat * 4, n_feat * 4, 4)
        self.se2 = Sty_layer(n_feat * 4, n_feat * 2, 4)
        self.se1 = Sty_layer(n_feat * 4, n_feat, 4)
        self.conv1 = nn.Conv2d(n_sequence, 1, 1, 1, "pad", 0, has_bias=True)

    def construct(self, x, flows, vec, mm, mean_):
        # inblock_fea,encoder_first_fea,encoder_second_fea,encoder_third_fea = prior
        mm = mm.squeeze(2)
        b,c,h,w = mm.shape
        mm1 = mm.mean(axis=1, keep_dims=True)
        mmap = self.conv1(mm) + mm1

        first_scale_inblock = self.inBlock(x)
        first_scale_encoder_first = self.encoder_first(first_scale_inblock) #torch.cat([first_scale_inblock,inblock_fea],dim=1)
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)  #torch.cat([first_scale_encoder_first,encoder_first_fea],dim=1)
        first_scale_encoder_third = self.encoder_third(first_scale_encoder_second)   #torch.cat([first_scale_encoder_second,encoder_second_fea],dim=1)
        first_scale_decoder_third = self.decoder_third(first_scale_encoder_third)  #torch.cat([first_scale_encoder_third,encoder_third_fea],dim=1)
        first_scale_decoder_third = self.se3(first_scale_decoder_third, vec) + first_scale_decoder_third
        out_ll = self.predict_ll(first_scale_decoder_third, flows, mmap, vec, first_scale_decoder_third.shape[-2:])
        
        first_scale_decoder_second = self.decoder_second(first_scale_decoder_third + first_scale_encoder_second)
        first_scale_decoder_second = self.se2(first_scale_decoder_second, vec) + first_scale_decoder_second
        out_l = self.predict_l(first_scale_decoder_second, flows, mmap, vec, first_scale_decoder_second.shape[-2:])
        out_l = [O.interpolate(out_ll[i], size=out_l[0].shape[-2:], mode='bilinear') + out_l[i] for i in range(len(out_l))]
        
        first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first)
        first_scale_decoder_first = self.se1(first_scale_decoder_first, vec) + first_scale_decoder_first
        out_ = self.predict(first_scale_decoder_first, flows, mmap, vec, first_scale_decoder_first.shape[-2:])
        out_ = [O.interpolate(out_l[i], size=out_[0].shape[-2:], mode='bilinear') + out_[i] for i in
                 range(len(out_))]
        
        first_scale_outBlock = self.outBlock(first_scale_decoder_first + first_scale_inblock)

        out = O.Split(axis=1, output_num=first_scale_outBlock.shape[1]//3)(first_scale_outBlock)
        out = list(out)
        fout = [out[i] + out_[i] for i in range(len(out))]
        mean_ = mean_.squeeze(2)
        fout = [o + mean_ for o in fout]

        return fout


class ResNet(nn.Cell):
    def __init__(self, n_inputs, n_outputs_e, n_outputs_r, blocks=[3,3,9,3], feats=32, loading=True, offset_network_path=None, extractor_path=None, halve=False):
        super(ResNet,self).__init__()
        self.UNet = UNet(in_channels=3, n_sequence=n_inputs, out_channels=3*(n_outputs_e+n_outputs_r), n_resblock=blocks, n_feat=feats,
                 kernel_size=3, n_outputs_e=n_outputs_e, n_outputs_r=n_outputs_r)
        if n_inputs % 2 !=0:
            tpp_inputs = n_inputs - 1
        else:
            tpp_inputs = n_inputs
        self.extractor = TPP(tpp_inputs, [2, 2, 2, 2], 64)

        self.refine = Refine_V(n_inputs*4, (n_outputs_e+n_outputs_r)*2)
        # self.flow_epoch = flow_epoch
        self.motion = ETR_motion_V2(offset_network_path is not None, 1, offset_network_path, offset_mode='se', vecdim=256)
        self.n_sequence = n_inputs
        if loading:
            print("Loading from ", extractor_path)
            ckpt=ms.load_checkpoint(extractor_path)
            nckpt={}
            for k in ckpt:
                nckpt["extractor."+k]=ckpt[k]
            ms.load_param_into_net(self.extractor,nckpt)

            for param in self.extractor.get_parameters():
                param.requires_grad = False
            # for param in self.motion.parameters():
            #     param.requires_grad = False


    def construct(self, images, epoch=None):
        images1 = O.stack(images, axis=2)
        mean_ = images1.mean((2,3,4), keep_dims=True)
        norm_x = images1 - mean_
        
        inputs = O.cat(images,1)

        if self.n_sequence % 2 !=0:
            tpp_inputs = O.cat(images[1:],1)
        else:
            tpp_inputs = inputs
        probas, vec1 = self.extractor(tpp_inputs, True)

        x = inputs
        b, c, h, w = x.shape
        norm_x = norm_x.view(b, c, h, w)
        imgs = x.view(b, self.n_sequence, 3, h, w)
        imgs_m = imgs.reshape(-1, 3, h, w)
        offsets = self.motion(imgs_m, vec1)

        delta = O.absolute(offsets[:, 2:, :, :] - offsets[:, :2, :, :])
        motion_map = O.sqrt(delta[:, 0:1, :, :].pow(2) + delta[:, 1:2, :, :].pow(2))
        motion_map = motion_map.view(b, self.n_sequence, 1, h, w)

        offsets = offsets.view(b, self.n_sequence, 4, h, w)
        offsets = offsets.view(b, self.n_sequence * 4, h, w)
        
        flows = self.refine(offsets, vec1)

        result = self.UNet(norm_x, flows, vec1, motion_map, mean_)

        return result, flows
