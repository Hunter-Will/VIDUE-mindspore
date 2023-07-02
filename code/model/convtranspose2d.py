import mindspore as ms
import mindspore.ops as O
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.common.initializer as init

class ConvTranspose2d(nn.Cell):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,dilation=1,padding=0,output_padding=0,has_bias=True):
        super(ConvTranspose2d,self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.dilation=dilation
        self.padding=padding
        self.output_padding=output_padding
        self.conv_t=O.Conv2DTranspose(in_channel,kernel_size,pad_mode="pad",pad=padding,stride=stride)
        self.weight=ms.Parameter(init.initializer("normal", (in_channel, out_channel, kernel_size, kernel_size)), name='weight')
        self.has_bias=has_bias
        if self.has_bias:
            self.bias=ms.Parameter(init.initializer("zeros", [out_channel]), name='bias')
        self.bias_add = P.BiasAdd()
        
    def construct(self,x):
        n,_,h,w=x.shape
        h_out=(h-1)*self.stride-2*self.padding+self.dilation*(self.kernel_size-1)+self.output_padding+1
        w_out=(w-1)*self.stride-2*self.padding+self.dilation*(self.kernel_size-1)+self.output_padding+1
        res=self.conv_t(x,self.weight,(n,self.out_channel,h_out,w_out))
        if self.has_bias:
            res=self.bias_add(res,self.bias)
        return res