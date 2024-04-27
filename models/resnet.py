import torch
import torch.nn as nn


# class Resnet18(nn.Module):
#     def __int__(self, block, image_chan, num_classes):
#         super(Resnet18, self).__int__()
#         # 224 * 224 * 3 -> 112 * 112 * 64
#         self.conv1 = nn.Conv2d(image_chan, out_channels=64, kernel_size=7, stride=2, padding=3)
#         # 112 * 112 * 64 -> 56 * 56 * 64
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # 56 * 56 * 64 -> 56 * 56 * 64
#         self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         # 56 * 56 * 64 -> 28 * 28 * 128
#         self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
#         self.conv3_2 = nn.Conv2d()

# 小于34层的Block
class BlockLess34(nn.Module):
    def __init__(self, in_chan, out_chan, stride=2, downsample=None, is_identity=True):
        super(BlockLess34, self).__init__()
        # 图片不缩小的时候stride为1
        # 其他图片缩小一半的情况是2
        self.ds = downsample
        self.is_identity = is_identity
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # 残差，但是需要注意一点的就是，当两层的通道数发生变化的时候，需要将残差进行下采样，修改至可以相加的样子

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        # x = self.relu(x)
        if self.ds is not None:
            identity = self.ds(identity)
        if self.is_identity:
            x = x + identity  # 不能使用+=，使用了梯度无法更新
        x = self.relu(x)

        return x


def layer_make(block, out_chan, repeat_num, stride, is_identity=True):
    """

    :param is_identity:
    :param stride:
    :param block:
    :param repeat_num: 重复块次数
    :param out_chan:
    :return:
    """
    downsample = None
    if stride == 2:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=int(out_chan / 2), out_channels=out_chan, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_chan)
        )

    if out_chan == 64:
        layers = [block(out_chan, out_chan, stride, is_identity=is_identity)]
    else:
        layers = [block(int(out_chan / 2), out_chan, stride, downsample, is_identity=is_identity)]
    for i in range(repeat_num - 1):
        layers.append(block(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, layers, image_chan, num_classes, is_identity=True):
        """
        默认图片大小为224*224
        :param layers: 18层次为[2,2,2,2], 34层次为[3,4,6,3],代表重复次数
        :param image_chan: 照片通道
        :param num_classes: 分类数量
        :param is_identity: 是否存在残差（用于对照实验）
        :return:
        """
        self.is_identity = is_identity
        super(ResNet, self).__init__()
        # 224 * 224 * 3 -> 112 * 112 * 64
        self.conv1 = nn.Conv2d(in_channels=image_chan, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # 112 * 112 * 64 -> 56 * 56 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = layer_make(BlockLess34, 64, layers[0], 1, self.is_identity)
        self.layer2 = layer_make(BlockLess34, 128, layers[1], 2, self.is_identity)
        self.layer3 = layer_make(BlockLess34, 256, layers[2], 2, self.is_identity)
        self.layer4 = layer_make(BlockLess34, 512, layers[3], 2, self.is_identity)

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgPool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmod(x)  # TODO 针对a_b_train的修改
        return x


def ResNet18(image_chan=3, num_classes=1000):
    return ResNet([2, 2, 2, 2], image_chan, num_classes, is_identity=True)


def ResNet34(image_chan=3, num_classes=1000):
    return ResNet([3, 4, 6, 3], image_chan, num_classes, is_identity=True)


def NoIdentityNet18(image_chan=3, num_classes=1000):
    return ResNet([2, 2, 2, 2], image_chan, num_classes, is_identity=False)


def NoIdentityNet34(image_chan=3, num_classes=1000):
    return ResNet([3, 4, 6, 3], image_chan, num_classes, is_identity=False)


if __name__ == "__main__":
    net = ResNet18()
    x = torch.randn(1, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)
