import torch.nn as nn
import torch


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=1)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1)


        self.convs1 = nn.Conv2d(2048, 512, 1)
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


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        conv3_3 = self.layer4(x)

        convs1 = self.convs1(conv3_3)
        decv3_1 = self.decv3_1(convs1)
        convs3_1 = self.convs3_1(self.pool2(input))
        convs3_2 = self.convs3_2(decv3_1)
        concat3 = torch.cat((convs3_1, convs3_2), 1)
        convs3_3 = self.convs3_3(concat3)
        convs3_4 = self.convs3_4(convs3_3)
        convs3_5 = self.convs3_4(convs3_4)
        convs3_6 = self.convs3_4(convs3_5)
        decv3_2 = self.decv3_2(convs3_6)

        return decv3_2
