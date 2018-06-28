import torch.nn as nn
import torch


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
       # self.features.append(nn.AvgPool2d(int(input_size/32)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.convs1 = nn.Conv2d(1280, 512, 1)
        self.decv3_1 = nn.ConvTranspose2d(512, 256, 8, stride=4, padding=2)
        self.convs3_1 = nn.Conv2d(3, 96, 9, padding=4)
        self.convs3_2 = nn.Conv2d(256, 64, 1)
        self.convs3_3 = nn.Conv2d(160, 64, 5, padding=2)
        self.convs3_4 = nn.Conv2d(64, 64, 5, padding=2)
        self.decv3_2 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        conv3_3 = self.features(x)
        convs1 = self.convs1(conv3_3)
        decv3_1 = self.decv3_1(convs1)
        convs3_1 = self.convs3_1(self.pool2(x))
        convs3_2 = self.convs3_2(decv3_1)
        concat3 = torch.cat((convs3_1, convs3_2), 1)
        convs3_3 = self.convs3_3(concat3)
        convs3_4 = self.convs3_4(convs3_3)
        convs3_5 = self.convs3_4(convs3_4)
        convs3_6 = self.convs3_4(convs3_5)
        decv3_2 = self.decv3_2(convs3_6)

        return decv3_2
