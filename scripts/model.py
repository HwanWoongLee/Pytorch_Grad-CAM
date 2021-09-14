import torch
import torch.nn as nn


def CBR2d(in_channels, out_channels, _kernal_size, _stride, _padding, _dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=_kernal_size, stride=_stride, padding=_padding, dilation=_dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class QVGGNetCAM(nn.Module):
    def __init__(self, num_classes=2):
        super(QVGGNetCAM, self).__init__()

        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()

        self.conv1_1 = CBR2d(3, 64, 3, 1, 1)
        self.conv1_2 = CBR2d(64, 64, 3, 1, 1)

        self.conv2_1 = CBR2d(64, 128, 3, 1, 1)
        self.conv2_2 = CBR2d(128, 128, 3, 1, 1)

        self.conv3_1 = CBR2d(128, 256, 3, 1, 1)
        self.conv3_2 = CBR2d(256, 256, 3, 1, 1)
        self.conv3_3 = CBR2d(256, 256, 3, 1, 1)

        self.conv4_1 = CBR2d(256, 512, 3, 1, 1)
        self.conv4_2 = CBR2d(512, 512, 3, 1, 1)
        self.conv4_3 = CBR2d(512, 512, 3, 1, 1)

        self.conv5_1 = CBR2d(512, 512, 3, 1, 1)
        self.conv5_2 = CBR2d(512, 512, 3, 1, 1)
        self.conv5_3 = CBR2d(512, 512, 3, 1, 1)

        # input image size 224x224x3
        self.feature = nn.Sequential(
            self.conv1_1,
            self.conv1_2,
            self.pooling,       # 112x112x128

            self.conv2_1,
            self.conv2_2,
            self.pooling,       # 56x56x256

            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.pooling,       # 28x28x256

            self.conv4_1,
            self.conv4_2,
            self.conv4_3,
            self.pooling,       # 14x14x512

            self.conv5_1,
            self.conv5_2,
            self.conv5_3,
            self.pooling,       # 7x7x512
        )

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.feature(x)
        x = self.avg(features)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, features


