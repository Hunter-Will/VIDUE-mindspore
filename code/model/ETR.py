import functools
import time

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O
import mindspore.common.initializer as init
import mindspore.numpy as mnp
from model.blocks import Sty_layer, CA_layer

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.set_data(init.initializer(init.HeUniform(),shape=m.weight.shape,dtype=m.weight.dtype))
        if hasattr(m.bias, 'data'):
            m.bias.set_data(init.initializer(init.Constant(0),shape=m.bias.shape,dtype=m.bias.dtype))
    elif classname.find('BatchNorm2d') != -1:
        m.weight.set_data(init.initializer(init.Normal(sigma=0.02,mean=1.0),shape=m.weight.shape,dtype=m.weight.dtype))
        m.bias.set_data(init.initializer(init.Constant(0),shape=m.bias.shape,dtype=m.bias.dtype))


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, use_batch_statistics=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_offset_quad(input_nc, nf, n_offset,offset_mode='quad', norm='batch'):
    norm_layer = get_norm_layer(norm_type=norm)

    net_offset = OffsetNet_quad(input_nc,nf,n_offset,offset_mode=offset_mode,norm_layer=norm_layer)

    net_offset.apply(weights_init)
    return net_offset

def define_offset_quad_v2(input_nc, nf, n_offset,offset_mode='quad', norm='batch', vecdim=256):

    norm_layer = get_norm_layer(norm_type=norm)

    net_offset = OffsetNet_quad_v2(input_nc,nf,n_offset,offset_mode=offset_mode,norm_layer=norm_layer, vecChannels=vecdim)

    net_offset.apply(weights_init)
    return net_offset

##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

class SpaceToDepth(nn.Cell):
    def __init__(self, block_size):
        super(SpaceToDepth,self).__init__()
        self.bs = block_size
    def construct(self, x):
        N, C, H, W = x.shape
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = O.transpose(x,(0, 3, 5, 1, 2, 4))  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

class ResnetBlock_woNorm(nn.Cell):

    def __init__(self, dim, use_bias):
        super(ResnetBlock_woNorm,self).__init__()
        padAndConv_1 = [
                nn.ReplicationPad2d(2),
                nn.Conv2d(dim, dim, kernel_size=5, pad_mode="pad", padding=0, has_bias=use_bias)]

        padAndConv_2 = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=5, pad_mode="pad", padding=0, has_bias=use_bias)]

        blocks = padAndConv_1 + [
            nn.ReLU()
        ]  + padAndConv_2 
        self.conv_block = nn.SequentialCell(blocks)

    def construct(self, x):
        x.shape
        out = x + self.conv_block(x)
        return out

def TriResblock(input_nc, use_bias=True):
    Res1 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    Res2 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    Res3 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    return nn.SequentialCell([Res1,Res2,Res3])



def conv_TriResblock(input_nc,out_nc,stride, use_bias=True):
    Relu = nn.ReLU()
    if stride==1:
        pad = nn.ReflectionPad2d(2)
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=1,padding=0,pad_mode="pad",has_bias=use_bias)
    elif stride==2:
        pad = nn.ReflectionPad2d((1,2,1,2))
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=2,padding=0,pad_mode="pad",has_bias=use_bias)
    tri_resblock = TriResblock(out_nc)
    return nn.SequentialCell([pad,conv,Relu,tri_resblock])

class Bottleneck(nn.Cell):
    def __init__(self,nChannels,kernel_size=3):
        super(Bottleneck,self).__init__()
        conv1 = nn.Conv2d(nChannels, nChannels*2, kernel_size=1, 
                                padding=0, pad_mode="pad", has_bias=True)
        lReLU1 = nn.LeakyReLU(0.2)
        conv2 = nn.Conv2d(nChannels*2, nChannels, kernel_size=kernel_size, 
                                padding=(kernel_size-1)//2, pad_mode="pad", has_bias=True)
        lReLU2 = nn.LeakyReLU(0.2)
        self.model = nn.SequentialCell([conv1,lReLU1,conv2,lReLU2])
    def construct(self,x):
        out = self.model(x)
        return out

class OffsetNet_quad(nn.Cell):
    # offset for Start and End Points, then calculate a quadratic function
    def __init__(self, input_nc, nf, n_offset, offset_mode='quad', norm_layer=nn.BatchNorm2d):
        super(OffsetNet_quad,self).__init__()
        self.input_nc = input_nc  #3
        self.nf = nf  #16
        self.n_offset = n_offset    # 15
        self.offset_mode = offset_mode
        if offset_mode == 'quad' or offset_mode == 'bilin':
            output_nc = 2 * 2
        elif offset_mode == 'lin':
            output_nc = 1 * 2
        else:
            output_nc = 2 * 2
        
        use_dropout = False
        use_bias=True

        self.pad_1 = nn.ReflectionPad2d((1,2,1,2))
        self.todepth = SpaceToDepth(block_size=2)
        self.conv_1 = conv_TriResblock(input_nc*4,nf,stride=1,use_bias=True)
        self.conv_2 = conv_TriResblock(nf,nf*2,stride=2,use_bias=True)
        self.conv_3 = conv_TriResblock(nf*2,nf*4,stride=2,use_bias=True)

        self.bottleneck_1 = Bottleneck(nf*4)
        self.uconv_1 = nn.Conv2dTranspose(nf*4, nf*2, kernel_size=4, stride=2, padding=1, 
                                        pad_mode="pad", has_bias=use_bias)

        self.bottleneck_2 = Bottleneck(nf*4)        
        self.uconv_2 = nn.Conv2dTranspose(nf*4, nf, kernel_size=4, stride=2, padding=1, 
                                        pad_mode="pad", has_bias=use_bias)
        self.bottleneck_3 = Bottleneck(nf*2)
        self.uconv_3 = nn.Conv2dTranspose(nf*2, nf*2, kernel_size=4, stride=2, padding=1, 
                                        pad_mode="pad", has_bias=use_bias)
        self.conv_out_0 = nn.Conv2d(nf*2,output_nc,kernel_size=5,stride=1,padding=2,pad_mode="pad",has_bias=use_bias)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.leaky2 = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def Quad_traj(self,offset10,offset12):
        B,C,H,W = offset10.shape
        N = self.n_offset//2
        t = mnp.arange(1,N,step=1,dtype=ms.float32)
        t = t/N
        t = t.view(-1,1,1,1)
        offset10 = offset10.view(B,1,2,H,W)
        offset12 = O.expand_dims(offset12,1)
        offset_12N = 0.5 * ((t + t**2)*offset12 - (t - t**2)*offset10)
        offset_10N = 0.5 * ((t + t**2)*offset10 - (t - t**2)*offset12)
        offset_12N = offset_12N.view(B,-1,H,W)   # b, (N-1)*c, h,w
        offset_10N = offset_10N.view(B,-1,H,W)

        return offset_10N,offset_12N

    def Bilinear_traj(self,offset10,offset12):
        B,C,H,W = offset10.shape
        N = self.n_offset//2
        t = mnp.arange(1,N,step=1,dtype=ms.float32)
        t = t/N
        t = t.view(-1,1,1,1)
        offset10 = offset10.view(B,1,2,H,W)
        offset12 = O.expand_dims(offset12,1)
        offset_12N = t * offset12
        offset_10N = t * offset10
        offset_12N = offset_12N.view(B,-1,H,W)
        offset_10N = offset_10N.view(B,-1,H,W)
        return offset_10N,offset_12N
    

    def construct(self, input):
        scale_0 = input
        B,N,H,W = input.shape
        scale_0_depth = self.todepth(scale_0)
        d_conv1 = self.conv_1(scale_0_depth)
        d_conv2 = self.conv_2(d_conv1)
        d_conv3 = self.conv_3(d_conv2)
        
        d_conv3 = self.bottleneck_1(d_conv3)
        u_conv1 = self.uconv_1(d_conv3)
        u_conv1 = self.leaky1(u_conv1) 
        u_conv1 = O.cat((u_conv1 , d_conv2),1)
        
        u_conv1 = self.bottleneck_2(u_conv1)
        u_conv2 = self.uconv_2(u_conv1)
        u_conv2 = self.leaky2(u_conv2)
        u_conv2 = O.cat((u_conv2 , d_conv1),1)

        u_conv2 = self.bottleneck_3(u_conv2)
        u_conv3 = self.uconv_3(u_conv2)

        out = self.conv_out_0(self.relu(u_conv3))
        # quadratic or bilinear
        if self.offset_mode == 'se':
            return out
        if self.offset_mode == 'quad' or self.offset_mode == 'bilin':
            offset_SPoint = out[:,:2,:,:]
            offset_EPoint = out[:,2:,:,:]
            if self.offset_mode == 'quad':
                offset_S_0, offset_0_E = self.Quad_traj(offset_SPoint,offset_EPoint)
            else:
                offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint,offset_EPoint)
        elif self.offset_mode == 'lin':
            # linear
            offset_SPoint = out
            offset_EPoint = 0 - out
            offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint,offset_EPoint)
        else:
            # return out[:, 2:, :, :] - out[:, :2, :, :]
            delta = O.absolute(out[:, 2:, :, :] - out[:, :2, :, :])
            return O.sqrt(delta[:,0:1,:,:].pow(2)+delta[:,1:2,:,:].pow(2))

        zeros = O.Zeros()((B,2,H,W),ms.float32)
        out = O.cat((offset_SPoint,offset_S_0,zeros,offset_0_E,offset_EPoint),1)  #b, c*(N-1+1+1+N-1+1),h,w
        return out


class OffsetNet_quad_v2(nn.Cell):
    # offset for Start and End Points, then calculate a quadratic function
    def __init__(self, input_nc, nf, n_offset, offset_mode='quad', norm_layer=nn.BatchNorm2d, vecChannels=256):
        super(OffsetNet_quad_v2, self).__init__()
        self.input_nc = input_nc  # 3
        self.nf = nf  # 16
        self.n_offset = n_offset  # 15
        self.offset_mode = offset_mode
        if offset_mode == 'quad' or offset_mode == 'bilin':
            output_nc = 2 * 2
        elif offset_mode == 'lin':
            output_nc = 1 * 2
        else:
            output_nc = 2 * 2

        use_dropout = False
        use_bias = True

        self.pad_1 = nn.ReflectionPad2d((1,2,1,2))
        self.todepth = SpaceToDepth(block_size=2)
        self.conv_1 = conv_TriResblock(input_nc * 4, nf, stride=1, use_bias=True)
        self.conv_2 = conv_TriResblock(nf, nf * 2, stride=2, use_bias=True)
        self.conv_3 = conv_TriResblock(nf * 2, nf * 4, stride=2, use_bias=True)

        self.bottleneck_1 = Bottleneck(nf * 4)
        self.uconv_1 = nn.Conv2dTranspose(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1,
                                          pad_mode="pad", has_bias=use_bias)

        self.bottleneck_2 = Bottleneck(nf * 4)
        self.uconv_2 = nn.Conv2dTranspose(nf * 4, nf, kernel_size=4, stride=2, padding=1,
                                          pad_mode="pad", has_bias=use_bias)
        self.bottleneck_3 = Bottleneck(nf * 2)
        self.uconv_3 = nn.Conv2dTranspose(nf * 2, nf * 2, kernel_size=4, stride=2, padding=1,
                                          pad_mode="pad", has_bias=use_bias)
        self.conv_out_0 = nn.Conv2d(nf * 2, output_nc, kernel_size=5, stride=1, padding=2, pad_mode="pad", has_bias=use_bias)

        if vecChannels == 256:
            reduc = 4
        else:
            reduc = 1
        self.use3 = CA_layer(vecChannels, nf * 2, reduc)
        self.use2 = CA_layer(vecChannels, nf * 4, reduc)
        self.use1 = CA_layer(vecChannels, nf * 4, reduc)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.leaky2 = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def Quad_traj(self, offset10, offset12):
        B, C, H, W = offset10.shape
        N = self.n_offset // 2
        t = mnp.arange(1,N,step=1,dtype=ms.float32)
        t = t / N
        t = t.view(-1, 1, 1, 1)
        offset10 = offset10.view(B, 1, 2, H, W)
        offset12 = O.expand_dims(offset12,1)
        offset_12N = 0.5 * ((t + t ** 2) * offset12 - (t - t ** 2) * offset10)
        offset_10N = 0.5 * ((t + t ** 2) * offset10 - (t - t ** 2) * offset12)
        offset_12N = offset_12N.view(B, -1, H, W)  # b, (N-1)*c, h,w
        offset_10N = offset_10N.view(B, -1, H, W)

        return offset_10N, offset_12N

    def Bilinear_traj(self, offset10, offset12):
        B, C, H, W = offset10.shape
        N = self.n_offset // 2
        t = mnp.arange(1,N,step=1,dtype=ms.float32)
        t = t / N
        t = t.view(-1, 1, 1, 1)
        offset10 = offset10.view(B, 1, 2, H, W)
        offset12 = O.expand_dims(offset12,1)
        offset_12N = t * offset12
        offset_10N = t * offset10
        offset_12N = offset_12N.view(B, -1, H, W)
        offset_10N = offset_10N.view(B, -1, H, W)
        return offset_10N, offset_12N

    def construct(self, input, vec):
        num = input.shape[0] // vec.shape[0]
        new_vec = []
        for i in range(vec.shape[0]):
            new_vec.append(mnp.tile(vec[i:i+1, :],(num,1)))
        vec = O.cat(new_vec,0)
        scale_0 = input
        B, N, H, W = input.shape
        scale_0_depth = self.todepth(scale_0)
        d_conv1 = self.conv_1(scale_0_depth)
        d_conv2 = self.conv_2(d_conv1)
        d_conv3 = self.conv_3(d_conv2)

        d_conv3 = self.bottleneck_1(d_conv3)
        d_conv3 = self.use1(d_conv3, vec)
        u_conv1 = self.uconv_1(d_conv3)
        u_conv1 = self.leaky1(u_conv1)
        u_conv1 = O.cat((u_conv1, d_conv2),1)

        u_conv1 = self.bottleneck_2(u_conv1)
        u_conv1 = self.use2(u_conv1, vec)
        u_conv2 = self.uconv_2(u_conv1)
        u_conv2 = self.leaky2(u_conv2)
        u_conv2 = O.cat((u_conv2, d_conv1),1)

        u_conv2 = self.bottleneck_3(u_conv2)
        u_conv2 = self.use3(u_conv2, vec)
        u_conv3 = self.uconv_3(u_conv2)

        out = self.conv_out_0(self.relu(u_conv3))
        # quadratic or bilinear
        if self.offset_mode == 'se':
            return out
        if self.offset_mode == 'quad' or self.offset_mode == 'bilin':
            offset_SPoint = out[:, :2, :, :]
            offset_EPoint = out[:, 2:, :, :]
            if self.offset_mode == 'quad':
                offset_S_0, offset_0_E = self.Quad_traj(offset_SPoint, offset_EPoint)
            else:
                offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint, offset_EPoint)
        elif self.offset_mode == 'lin':
            # linear
            offset_SPoint = out
            offset_EPoint = 0 - out
            offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint, offset_EPoint)
        else:
            # return out[:, 2:, :, :] - out[:, :2, :, :]
            delta = O.absolute(out[:, 2:, :, :] - out[:, :2, :, :])
            return O.sqrt(delta[:, 0:1, :, :].pow(2) + delta[:, 1:2, :, :].pow(2))

        zeros = O.Zeros()((B,2,H,W),ms.float32)
        out2 = O.cat((offset_SPoint, offset_S_0, zeros, offset_0_E, offset_EPoint),1)  # b, c*(N-1+1+1+N-1+1),h,w
        return out, out2

class ETR_motion(nn.Cell):
    def __init__(self, pre_trained, n_GPUs, offset_network_path, offset_mode='none'):
        super(ETR_motion,self).__init__()
        self.offset_net = define_offset_quad(input_nc=3, nf=16, n_offset=15, norm='batch', offset_mode=offset_mode)
        self.n_GPUs = n_GPUs
        self.n_offset = 15
        if pre_trained:
            ms.load_param_into_net(self.offset_net,ms.load_checkpoint(offset_network_path))
            print('Loading Offset pretrain model from {}'.format(offset_network_path))
    def construct(self, img_in):

        offset = self.offset_net(img_in)   ##b, c*(1+N-1+1+N-1+1),h,w

        return offset


class ETR_motion_V2(nn.Cell):
    def __init__(self, pre_trained, n_GPUs, offset_network_path, offset_mode='none', vecdim=256):
        super(ETR_motion_V2,self).__init__()
        self.offset_net = define_offset_quad_v2(input_nc=3, nf=16, n_offset=15, norm='batch', offset_mode=offset_mode, vecdim=vecdim)
        self.n_GPUs = n_GPUs
        self.n_offset = 15
        if pre_trained:
            pretrained_model = ms.load_checkpoint(offset_network_path)
            pretrained_dict = pretrained_model
            model_dict = self.offset_net.parameters_dict()
            for k, v in pretrained_dict.items():
                if k not in model_dict:
                    print("Not in model", k)
                elif v.shape != model_dict[k].shape:
                    print("Mismatch shape:", k)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                                  k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            ms.load_param_into_net(self.offset_net,model_dict)
            print('Loading Offset pretrain model from {}'.format(offset_network_path))

    def construct(self, img_in, vec):

        offset = self.offset_net(img_in, vec)   ##b, c*(1+N-1+1+N-1+1),h,w

        return offset

