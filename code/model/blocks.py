import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as O
import mindspore.common.initializer as init
###############################
# common
###############################

class SEBlock(nn.Cell):
    def __init__(self, input_dim, reduction):
        super(SEBlock,self).__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = O.AdaptiveAvgPool2D(1)
        self.fc = nn.SequentialCell([
            nn.Dense(input_dim, mid),
            nn.ReLU(),
            nn.Dense(mid, input_dim),
            nn.Sigmoid()
        ])

    def construct(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class ResBlock(nn.Cell):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bn=False, se=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, pad_mode="pad", has_bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, pad_mode="pad", has_bias=True)
        self.relu = nn.LeakyReLU(0.1)
        if se:
            self.se = SEBlock(planes, 4)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes,use_batch_statistics=True)
            self.bn2 = nn.BatchNorm2d(planes,use_batch_statistics=True)
        self.se_ = se
        self.bn_ = bn
        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, pad_mode="pad", padding=0, has_bias=True)

    def construct(self, x):
        residual = x
        if self.bn_:
            out = self.relu(self.bn1(self.conv1(x)))
        else:
            out = self.relu(self.conv1(x))
        if self.bn_:
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(out)
        #print(out)
        if self.se_:
            out = self.se(out)
        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out

class CA_layer(nn.Cell):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.SequentialCell([
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, padding=0, pad_mode="pad", has_bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, padding=0, pad_mode="pad", has_bias=False),
            nn.Sigmoid()
        ])

    def construct(self, x,vec):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(vec[:, :, None, None])

        return x * att

class ChanNorm(nn.Cell):
    def __init__(self, dim, eps = 1e-5):
        super(ChanNorm,self).__init__()
        self.eps = eps
        self.g = ms.Parameter(O.Ones()((1, dim, 1, 1), ms.float32))
        self.b = ms.Parameter(O.Zeros()((1, dim, 1, 1), ms.float32))

    def construct(self, x):
        var = x.var(axis = 1, keepdims = True)
        mean = x.mean(axis = 1, keep_dims = True)
        return ((x - mean) / (var + self.eps).sqrt()) * self.g + self.b

class Conv2DMod(nn.Cell):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super(Conv2DMod,self).__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = ms.Parameter(O.normal((out_chan, in_chan, kernel, kernel),0,1))
        self.eps = eps
        self.weight.set_data(init.initializer(init.HeNormal(negative_slope=0, mode='fan_in', nonlinearity='leaky_relu'),
                                              shape=self.weight.shape,dtype=self.weight.dtype))

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def construct(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = O.rsqrt((weights ** 2).sum(axis=(2, 3, 4), keepdims=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, d2, d3, d4 = weights.shape
        weights = weights.reshape(b * self.filters, d2, d3, d4)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = O.conv2d(x, weights, pad_mode="pad", padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class Sty_layer(nn.Cell):
    def __init__(self, channels_in, channels_out, reduction):
        super(Sty_layer, self).__init__()
        self.norm = ChanNorm(channels_out)
        self.conv_du = nn.SequentialCell([
            nn.Dense(channels_in, channels_in//reduction, has_bias=False),
            nn.LeakyReLU(0.1),
            nn.Dense(channels_in // reduction, channels_out, has_bias=False),
            # nn.Sigmoid()
        ])
        self.conv = Conv2DMod(channels_out, channels_out, 3)

    def construct(self, x, vec):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        norm_x = self.norm(x)
        sty = self.conv_du(vec)
        out = self.conv(norm_x, sty)
        return out + x