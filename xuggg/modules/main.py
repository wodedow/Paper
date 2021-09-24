import torch
import torch.nn as nn
from torchvision import models
from modules.conpcon import ContentPartialConv2d
from modules.attention import AttentionModule, AttentionModuleMask
from modules.PConvLayer import PConvLayer
from modules.involution import Involution


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class VGG16FeatureExtractor(nn.Module):
    '''
    Using pretrained VGG16 to caculate the loss of Pertual and Style
    '''
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i + 1}').parameters():
                param.requires_grad = False

    def forward(self, image):
        result = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i + 1}')
            result.append(func(result[-1]))

        return result[1:]


class Bottleneck(nn.Module):
    '''
    The sequences of Tensors:
        x - conv1-bn-relu - conv3-bn-relu - conv1-bn - cat-relu
    The changes of Channels:
        inplanes-planes-planes-planes*4-inplanes
    '''
    expansion = 4  # inplanes = planes * 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.invo1 = Involution(planes * 4, kernel_size=3, stride=stride)
        self.bn4 = nn.BatchNorm2d(planes * 4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.invo1(out)
        out = self.bn4(out)
        out += residual
        out = self.relu(out)
        return out


class ContentEnCoder(nn.Module):
    def __init__(self):
        super(ContentEnCoder, self).__init__()
        self.conpconv1 = ContentPartialConv2d(in_channels=3,
                                              out_channels=64,
                                              kernel_size=7,
                                              stride=2,
                                              padding=3,
                                              multi_channel=True,
                                              bias=False,
                                              size=128)
        self.conpconv2 = ContentPartialConv2d(in_channels=64,
                                              out_channels=64,
                                              kernel_size=7,
                                              stride=1,
                                              padding=3,
                                              multi_channel=True,
                                              bias=False,
                                              size=128)
        for i in range(3):
            conpconv = f'conpconv{i+3}'
            block = ContentPartialConv2d(in_channels=64,
                                         out_channels=64,
                                         kernel_size=7,
                                         stride=1,
                                         padding=3,
                                         multi_channel=True,
                                         bias=False,
                                         size=128)
            setattr(self, conpconv, block)

        self.conpconv6 = ContentPartialConv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              multi_channel=True,
                                              bias=False,
                                              size=64)
        self.conpconv7 = ContentPartialConv2d(in_channels=128,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              multi_channel=True,
                                              bias=False,
                                              size=64)
        self.pconv1 = PConvLayer(128, 256, activ='relu', sample='down-2')
        self.pconv2 = PConvLayer(256, 256, activ='relu')
        self.pconv3 = PConvLayer(256, 512, activ='relu', sample='down-2')
        for i in range(3):
            pconv = f'pconv{i+4}'
            block = PConvLayer(512, 512, activ='relu', sample='down-1')
            setattr(self, pconv, block)

    def forward(self, image, mask):
        outputs = [image]
        masks = [mask]
        outputs_ = [image]
        masks_ = [mask]

        x1, m1, _, _ = self.conpconv1(image, mask)

        x1, m1, x2, m2 = self.conpconv2(x1, m1)  # Size(bs, 64, 128, 128)
        outputs.append(x1)
        masks.append(m1)
        outputs_.append(x2)
        masks_.append(m2)

        for i in range(4):
            conpconv = f'conpconv{i+3}'
            x1, m1, _, _ = getattr(self, conpconv)(x1, m1)

        x1, m1, x2, m2 = self.conpconv7(x1, m1)  # Size(bs, 128, 64, 64)
        outputs.append(x1)
        masks.append(m1)
        outputs_.append(x2)
        masks_.append(m2)

        for i in range(6):
            pconv = f'pconv{i+1}'
            x1, m1 = getattr(self, pconv)(x1, m1)
            if i in [1, 3, 4, 5]:
                outputs.append(x1)
                masks.append(m1)

        return outputs, masks, outputs_, masks_


class ContentDeCoder(nn.Module):
    def __init__(self, transfer=True):
        super(ContentDeCoder, self).__init__()
        self.transfer = transfer
        self.pconv1 = PConvLayer(1024, 512, sample='down-0', activ='relu')
        self.pconv2 = PConvLayer(1024, 256, activ='leaky', deconv=True)
        self.pconv3 = PConvLayer(512, 128, activ='leaky', deconv=True)
        self.pconv4 = PConvLayer(256, 64, activ='leaky', deconv=True)
        self.pconv5 = PConvLayer(128, 32, activ='leaky', deconv=True)
        self.attention = AttentionModule(256)
        if self.transfer:
            self.attention_transfer1 = AttentionModuleMask(128)
            self.attention_transfer2 = AttentionModuleMask(64)

    def forward(self, inputs, masks, alter=0.0):
        x1 = torch.cat([inputs[-2], inputs[-1]], dim=1)
        m1 = torch.cat([masks[-2], masks[-1]], dim=1)
        x1, m1 = self.pconv1(x1, m1)  # Size(bs, 512, 16, 16)

        x1 = torch.cat([x1, inputs[-3]], dim=1)
        m1 = torch.cat([m1, masks[-3]], dim=1)
        x1, m1 = self.pconv2(x1, m1)  # Size(bs, 256, 32, 32)

        x1 = self.attention(x1)  # Size(bs, 256, 32, 32)

        x1 = torch.cat([x1, inputs[-4]], dim=1)
        m1 = torch.cat([m1, masks[-4]], dim=1)
        x1, m1 = self.pconv3(x1, m1)  # Size(bs, 128, 64, 64)

        if self.transfer:
            x1 = self.attention_transfer1(x1, masks[-5],
                                          alter[-1])  # Size(bs, 128, 64, 64)

        x1 = torch.cat([x1, inputs[-5]], dim=1)
        m1 = torch.cat([m1, masks[-5]], dim=1)
        x1, m1 = self.pconv4(x1, m1)  # Size(bs, 64, 128, 128)

        if self.transfer:
            x1 = self.attention_transfer2(x1, masks[-6],
                                          alter[-2])  # Size(bs, 64, 128, 128)

        x1 = torch.cat([x1, inputs[-6]], dim=1)
        m1 = torch.cat([m1, masks[-6]], dim=1)
        x1, m1 = self.pconv5(x1, m1)  # Size(bs, 32, 256, 256)

        return x1, m1


class Main(BaseNetwork):
    def __init__(self):
        super(Main, self).__init__()
        self.encoder = ContentEnCoder()
        self.decoder = ContentDeCoder(transfer=True)
        self.decoder_sub = ContentDeCoder(transfer=False)

        self.bottleneck1 = Bottleneck(32, 8)
        self.bottleneck2 = Bottleneck(32, 8)

        self.pconv1 = PConvLayer(35, 32, activ='relu')
        self.pconv2 = PConvLayer(35, 32, activ='relu')

        self.res = nn.Conv2d(in_channels=64,
                             out_channels=3,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=True)

    def forward(self, image, mask):
        outputs, masks, outputs_, masks_ = self.encoder(image, mask)
        # print("process: encoder")

        x1, m1 = self.decoder(outputs, masks, alter=outputs_)
        x1 = self.bottleneck1(x1)
        x1 = torch.cat([x1, image], dim=1)
        m1 = torch.cat([m1, mask], dim=1)
        x1, _ = self.pconv1(x1, m1)
        # print("process: encoder")

        outputs_ = outputs_ + outputs[3:]
        masks_ = masks_ + masks[3:]
        x2, m2 = self.decoder_sub(outputs_, masks_)
        x2 = self.bottleneck2(x2)
        x2 = torch.cat([x2, image], dim=1)
        m2 = torch.cat([m2, mask], dim=1)
        x2, _ = self.pconv2(x2, m2)
        # print("process: encoder_sub")

        x3 = torch.cat([x2, x2], dim=1)
        result = self.res(x3)

        return result

    def train(self, mode=True, finetune=False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
