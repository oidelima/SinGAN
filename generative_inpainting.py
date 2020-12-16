from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.models import *
from SinGAN.imresize import *
import SinGAN.functions as functions
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import random
from torchvision.utils import save_image

from tools import extract_image_patches, flow_to_image, \
    reduce_mean, reduce_sum, default_loader, same_padding


class Generator(nn.Module):
    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow
        

class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, x, mask):
        # For indicating the boundaries of images
  
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1


class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum*2, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=self.use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2, offset_flow
        

class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(4*self.rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if self.use_cuda:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*4, mode='nearest')

        return y, flow
    
    
def test_contextual_attention(args):
    import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    def float_to_uint8(img):
        img = img * 255
        return img.astype('uint8')

    rate = 2
    stride = 1
    grid = rate*stride

    b = default_loader(args.imageA)
    w, h = b.size
    b = b.resize((w//grid*grid//2, h//grid*grid//2), Image.ANTIALIAS)
    # b = b.resize((w//grid*grid, h//grid*grid), Image.ANTIALIAS)
    print('Size of imageA: {}'.format(b.size))

    f = default_loader(args.imageB)
    w, h = f.size
    f = f.resize((w//grid*grid, h//grid*grid), Image.ANTIALIAS)
    print('Size of imageB: {}'.format(f.size))

    f, b = transforms.ToTensor()(f), transforms.ToTensor()(b)
    f, b = f.unsqueeze(0), b.unsqueeze(0)
    if torch.cuda.is_available():
        f, b = f.cuda(), b.cuda()

    contextual_attention = ContextualAttention(ksize=3, stride=stride, rate=rate, fuse=True)

    if torch.cuda.is_available():
        contextual_attention = contextual_attention.cuda()

    yt, flow_t = contextual_attention(f, b)
    vutils.save_image(yt, 'vutils' + args.imageOut, normalize=True)
    vutils.save_image(flow_t, 'flow' + args.imageOut, normalize=True)
    # y = tensor_img_to_npimg(yt.cpu()[0])
    # flow = tensor_img_to_npimg(flow_t.cpu()[0])
    # cv2.imwrite('flow' + args.imageOut, flow_t)


class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--run_name', help='name of experimental run', required=True)
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mask_dir', help='input mask dir', default='Input/masks')
    parser.add_argument('--mask_name', help='input mask name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--crop_size', type=int, default=250)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--random_crop', action='store_true', help='enables random crop during training')
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--fixed_eye_loc', nargs='+', type=int)
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
    parser.add_argument('--checkpoint_path', type=str, default='')
    
    config = {'dataset_name': 'imagenet', 'data_with_subfolder': True, 
          'train_data_path': '/media/ouc/4T_A/datasets/ImageNet/ILSVRC2012_img_train/', 
          'val_data_path': None, 'resume': None, 'batch_size': 48, 'image_shape': [256, 256, 3],
          'mask_shape': [128, 128], 'mask_batch_same': True, 'max_delta_shape': [32, 32], 
          'margin': [0, 0], 'discounted_mask': True, 'spatial_discounting_gamma': 0.9, 
          'random_crop': True, 'mask_type': 'hole', 'mosaic_unit_size': 12, 
          'expname': 'benchmark', 'cuda': True, 'gpu_ids': [0], 'num_workers': 4, 
          'lr': 0.0001, 'beta1': 0.5, 'beta2': 0.9, 'n_critic': 5, 'niter': 500000, 
          'print_iter': 100, 'viz_iter': 1000, 'viz_max_out': 16, 'snapshot_save_iter': 5000, 
          'coarse_l1_alpha': 1.2, 'l1_loss_alpha': 1.2, 'ae_loss_alpha': 1.2, 
          'global_wgan_loss_alpha': 1.0, 'gan_loss_alpha': 0.001, 'wgan_gp_lambda': 10, 
          'netG': {'input_dim': 3, 'ngf': 32}, 'netD': {'input_dim': 3, 'ndf': 64}}  
    
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    dir2save = functions.generate_dir2save(opt)
    

    
    def load_ckpt(ckpt_name, models, optimizers=None):
        ckpt_dict = torch.load(ckpt_name)
        for prefix, model in models:
            assert isinstance(model, nn.Module)
            model.load_state_dict(ckpt_dict[prefix], strict=False)
        if optimizers is not None:
            for prefix, optimizer in optimizers:
                optimizer.load_state_dict(ckpt_dict[prefix])
        return ckpt_dict['n_iter']
    

    if (os.path.exists(dir2save)) and opt.run_name != "test":
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
          
        # MEAN = [0.485, 0.456, 0.406]
        # STD = [0.229, 0.224, 0.225]
        
       
        # LAMBDA_DICT = {
        #     'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
        
        cuda = config['cuda']
        device_ids = config['gpu_ids']
        if cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
            device_ids = list(range(len(device_ids)))
            config['gpu_ids'] = device_ids
            cudnn.benchmark = True
        # if args.seed is None:
        #     args.seed = random.randint(1, 10000)
        # print("Random seed: {}".format(args.seed))
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # if cuda:
        #     torch.cuda.manual_seed_all(args.seed)
    
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        # torch.cuda.set_device(opt.device)

        real_ = functions.read_image(opt)
        real = imresize(real_,opt.scale1,opt)
        orig_size = (real.shape[2], real.shape[3])
        # real = imresize_to_shape(real_,(256, 256),opt)
        
        

        
        mask = functions.read_mask(opt,"Input/body_masks",opt.mask_name) 
        # constraint = functions.read_mask(opt, "Input/constraints", opt.mask_name)
        # mask_ = functions.read_mask(opt) 
        # mask_ = imresize_mask_to_shape(mask_, (256, 256), opt)

       # Loading image source for mask and resizing so that biggest dimension is opt.patch_size 
        new_dim = (mask.size()[3], mask.size()[2])
        mask_source = functions.read_image(opt, "Input/mask_sources", opt.mask_name[:-3]+"jpg", size=new_dim)
        
        
        constraint_ = functions.generate_eye_mask(opt, mask, 0, opt.fixed_eye_loc)
        constraint = constraint_ * mask #* mask_source
        
        
        height, width = real.shape[2], real.shape[3]
        scale = 256/min(height, width)
        size = (int(height*scale), int(width*scale))
  
     
        img_transform = transforms.Compose(
            [   transforms.Resize(size),
                transforms.ToTensor()])
        mask_transform = transforms.Compose(
            [ transforms.Resize(size=(int(mask.shape[2]*scale), int(mask.shape[3]*scale)), interpolation=0),transforms.ToTensor()])
        
       

        netG = Generator(config['netG'], cuda, device_ids).to(opt.device)
        # load_ckpt("weights.pth", [('model', model)])    
        # model.eval()


        real = Image.open("Input/Images/" + opt.input_name)
        
        netG.load_state_dict(torch.load("gen_00430000.pt"))
        real_ = img_transform(real).unsqueeze(0).to(opt.device)
        real_ = real_*2 -1
        
     
        # mask_source = mask_source*2 - 1
       
        real_cropped = real_[:,:,int(size[0]//2) - 128:int(size[0]//2) + 128, int(size[1]//2) - 128: int(size[1]//2) + 128]
        
        constraint_ = torch.nn.functional.interpolate(constraint, size=(int(mask.shape[2]*scale), int(mask.shape[3]*scale)), mode='nearest')
        mask_source = torch.nn.functional.interpolate(mask_source, size=(int(mask.shape[2]*scale), int(mask.shape[3]*scale)))

        mask = Image.open("Input/body_masks/" + opt.mask_name)
        mask_ = mask_transform(mask).unsqueeze(0).to(opt.device)

        
 
        for i in range(20):
            
            im_height, im_width = real_cropped.size()[2], real_cropped.size()[3] 
            mask_height, mask_width = mask_.size()[2], mask_.size()[3] 
            position = (np.random.randint(im_height - mask_height), np.random.randint(im_width - mask_width))
            _, _, constraint_ind, mask_ind, constraint_filled = functions.gen_fake(real_cropped, real_cropped, mask_[:,0:1,:,:], constraint_, mask_source, opt, position=position)
            
            # mask = mask.repeat(1,3,1,1)
            x = real_cropped*(1-mask_ind).to(opt.device) #+ ((constraint_filled).to(opt.device))
            
            # with torch.no_grad():
            #     out, out_mask = netG(real*mask.to(opt.device), mask.to(opt.device))
            
            x1, x2, offset_flow = netG(x.to(opt.device), mask_ind.to(opt.device))
            
            inpainted_result, _, constraint_ind, mask_ind, constraint_filled = functions.gen_fake(real_cropped, x2, mask_[:,0:1,:,:], constraint_, mask_source, opt, position_background=position)
            # inpainted_result = x2 * (mask_ind.to(opt.device)-constraint_ind.to(opt.device)) + x * (1. - mask_ind.to(opt.device)) + constraint_filled.to(opt.device)
     

            output_full = real_.clone()
            border_ind_full =  real_.clone()
            mask_ind_full = torch.zeros_like(real_)
            output_full[:,:,int(size[0]//2) - 128:int(size[0]//2) + 128, int(size[1]//2) - 128: int(size[1]//2) + 128]  = inpainted_result
            mask_ind_full[:,:,int(size[0]//2) - 128:int(size[0]//2) + 128, int(size[1]//2) - 128: int(size[1]//2) + 128]  =mask_ind
            
            def post(image):
                # image = image.transpose(1, 3)
                image = (image  + 1)/2
                # image = image.transpose(1, 3)
                image = torch.nn.functional.interpolate(image, size=orig_size, mode='bicubic')
                return image
        
            output_full = post(output_full)
            # save_image(output_full, 'test.png' )
            # sd
            mask_ind_full = torch.nn.functional.interpolate(mask_ind_full, size=orig_size, mode='nearest')

      
            dilated = torch.tensor(scipy.ndimage.morphology.binary_dilation(np.asarray(mask_ind_full.cpu()),iterations=1)).to(mask_ind_full)
            border = (dilated-mask_ind_full).to(opt.device).round()[:,0:1,:,:]
        
            
            border_colored= torch.zeros([1,3,orig_size[0], orig_size[1]]).to(opt.device)
            border_colored[:,0:1,:,:] = border*255 
            border_colored[:,1:,:,:] = -border*255 
            border_ind_full = output_full* (1-border) +border_colored
            

            dir2save = '%s/RandomSamples/%s/gen_inpainting/%s' % (opt.out, opt.input_name[:-4], opt.run_name)
            
            try:
                os.makedirs(dir2save + "/fake")
                os.makedirs(dir2save + "/border_ind")
                # os.makedirs(dir2save + "/background")
                # os.makedirs(dir2save + "/mask")
                # os.makedirs(dir2save + "/constraint")
                os.makedirs(dir2save + "/mask_ind")

            except OSError:
                pass


            save_image(output_full, '%s/%s/%d.png' % (dir2save, "fake", i))
            save_image(border_ind_full, '%s/%s/%d.png' % (dir2save, "border_ind", i))
            save_image(mask_ind_full, '%s/%s/%d.png' % (dir2save, "mask_ind", i))
            # plt.imsave('%s/%s/%d.png' % (dir2save, "fake", i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "background", i), np.asarray(opt_img_), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "mask", i), functions.convert_image_np(fake_ind.detach()), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "mask_ind", i), functions.convert_image_np(mask_ind.detach()),  cmap="gray")
            # plt.imsave('%s/%s/%d.png' % (dir2save, "constraint", i), functions.convert_image_np(constraint_filled.detach()), vmin=0,vmax=1)

                    
              
        


    
