import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch


def feature_layer():
    layers = []
    pool1 = ['4', '9', '16']
    pool2 = ['23', '30']
    vgg16 = models.vgg16(pretrained=True).features
    for name, layer in vgg16._modules.items():
        if isinstance(layer, nn.Conv2d):
            layers += [layer, nn.Dropout2d(0.5), nn.PReLU()]
        elif name in pool1:
            layers += [layer]
        elif name == pool2[0]:
            layers += [nn.MaxPool2d(2, 1, 1)]
        elif name == pool2[1]:
            layers += [nn.MaxPool2d(2, 1, 0)]
        else:
            continue
    features = nn.Sequential(*layers)
    #feat3 = features[0:24]
    return features


class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.features = feature_layer()
        self.feat = nn.Sequential(
            nn.Conv2d(160, 64, 5, 1, 2),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ConvTranspose2d(64, 3, 4, 2, 1)
        )
        self.decv3_1 = nn.ConvTranspose2d(512, 256, 8, 4, 2)
        self.convs3_1 = nn.Conv2d(3, 96, 9, 1, 4)
        self.convs3_2 = nn.Conv2d(256, 64, 1)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self._initialize_weights()


    def forward(self, x):
        conv3_3 = self.features(x)
        decv3_1 = self.decv3_1(conv3_3)
        convs3_1 = self.convs3_1(self.pool(x))
        convs3_2 = self.convs3_2(decv3_1)
        concat3 = torch.cat((convs3_1, convs3_2), 1)
        output = self.feat(concat3)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if m not in self.features:
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.normal_(m.weight, mean=0, std=0.001)
                    torch.nn.init.constant_(m.bias, 0.1)
