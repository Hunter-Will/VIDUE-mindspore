import math
import numpy as np
import importlib

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O
from model import blocks
from model.ETR import ETR_motion

def make_model(args):
    return TPP(n_inputs=args.n_sequence, blocks=args.n_resblock, feats=args.n_feat, xN=args.n_outputs, offset_network_path=args.offset_network_path, halve=args.halve)

class Exractor(nn.Cell):

    def __init__(self, in_channels=3, n_sequence=3, n_resblock=[3,3,9,3], n_feat=32, kernel_size=3, xN=8):
        super(Exractor, self).__init__()
        InBlock = []

        InBlock.extend([nn.SequentialCell([
            nn.Conv2d(in_channels * n_sequence, n_feat, kernel_size=7, stride=1,
                      padding=7 // 2, pad_mode="pad", has_bias=True),
            nn.BatchNorm2d(n_feat,use_batch_statistics=True),
            nn.LeakyReLU(0.1)]
        )])

        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1, bn=True, se=False)
                        for _ in range(n_resblock[0])])

        # encoder1
        Encoder_first = [nn.SequentialCell([
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, pad_mode="pad", has_bias=True),
            nn.BatchNorm2d(n_feat*2,use_batch_statistics=True),
            nn.LeakyReLU(0.01)]
        )]
        Encoder_first.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1, bn=True, se=False)
                              for _ in range(n_resblock[1])])
        # encoder2
        Encoder_second = [nn.SequentialCell([
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, pad_mode="pad", has_bias=True),
            nn.BatchNorm2d(n_feat*4,use_batch_statistics=True),
            nn.LeakyReLU(0.01)]
        )]
        Encoder_second.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, bn=True, se=False)
                               for _ in range(n_resblock[2])])

        # encoder3
        Encoder_third = [nn.SequentialCell([
            nn.Conv2d(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, pad_mode="pad", has_bias=True),
            nn.BatchNorm2d(n_feat * 4,use_batch_statistics=True),
            nn.LeakyReLU(0.01)]
        )]
        Encoder_third.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, bn=True, se=False)
                               for _ in range(n_resblock[3])])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.inBlock = nn.SequentialCell(InBlock)
        self.encoder_first = nn.SequentialCell(Encoder_first)
        self.encoder_second = nn.SequentialCell(Encoder_second)
        self.encoder_third = nn.SequentialCell(Encoder_third)
        self.mlp = nn.SequentialCell([
            nn.Dense(n_feat * 4, n_feat * 4 * 4),
            nn.LeakyReLU(0.01),
            nn.Dense(n_feat * 4 * 4, n_feat * 4)
        ])
        self.conv1 = nn.Conv2d(n_sequence, 1, 1, 1, padding=0, pad_mode="pad", has_bias=True)
        self.num_classes = xN
        self.n_sequence = n_sequence
        self.sigmoid = nn.Sigmoid()

    def construct(self, x, offsets):
        b, t, c, h, w = offsets.shape
        mmap = offsets.view(b, self.n_sequence, h, w).mean(axis=1, keep_dims=True)
        mmap = self.conv1(offsets.view(b, self.n_sequence, h, w)) + mmap
        mmap = self.sigmoid(mmap)
        first_scale_inblock = self.inBlock(x * mmap)
        first_scale_encoder_first = self.encoder_first(first_scale_inblock * mmap)

        mmap_lv1 = O.interpolate(mmap, size=first_scale_encoder_first.shape[-2:], mode='bilinear')
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first * mmap_lv1)
        mmap_lv2 = O.interpolate(mmap_lv1, size=first_scale_encoder_second.shape[-2:], mode='bilinear')
        first_scale_encoder_third = self.encoder_third(first_scale_encoder_second * mmap_lv2)
        
        vec = self.avgpool(first_scale_encoder_third).squeeze(-1).squeeze(-1)
        vec = self.mlp(vec)
        return vec  #, logits, probas


class TPP(nn.Cell):
    def __init__(self, n_inputs, blocks=[3,3,9,3], feats=32, xN=8, halve=False, offset_network_path=None, fix=True):
        super(TPP,self).__init__()
        if halve:
            xN = xN // 2
            print("Halving xN!")
        self.extractor = Exractor(in_channels=3, n_sequence=n_inputs, n_resblock=blocks, n_feat=feats,
                 kernel_size=3, xN=xN)
        self.n_sequence = n_inputs
        self.motion = ETR_motion(offset_network_path is not None, 1, offset_network_path)
        if fix:
            for param in self.motion.get_parameters():
                param.requires_grad = False


    def construct(self, images, Visual=False):
        x = images
        b, c, h, w = x.shape
        imgs = x.view(b, self.n_sequence, 3, h, w)
        imgs_m = imgs.reshape(-1, 3, h, w)
        offsets = self.motion(imgs_m)
        offsets = offsets.view(b, self.n_sequence, 1, h, w)

        # compute query features
        vec = self.extractor(images, offsets)  # , logits, probas

        embedding = []
        if Visual:
            return embedding, vec
        vec = O.L2Normalize(1,1e-12)(vec)
        return embedding, vec


