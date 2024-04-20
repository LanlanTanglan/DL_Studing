import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.a_b_dataset import ABDataset
from models.resnet import ResNet18

"""
超参数
"""
learning_rate = 1e-4
batch_size = 20
epochs = 100
on_gpu = True

# 变换
default_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 载入数据集
train_data = ABDataset("datasets\\apple_banana_datasets\\train", default_transform)
val_data = ABDataset("datasets\\apple_banana_datasets\\val", default_transform)
test_data = ABDataset("datasets\\apple_banana_datasets\\test", default_transform)

# 分类数量标签
labels = ['apple', 'banana']

# 载入数据集加载器
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# 载入模型
resnet18 = ResNet18(num_classes=len(labels) - 1)
resnet18 = resnet18.cuda()  # 采用cuda
# 损失函数 随便选个先试试，因为是二分类任务
loss_fn = nn.BCELoss()
# 优化器 随便选个先试试
optimizer = torch.optim.SGD(resnet18.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fun, opt_fun):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        if on_gpu:
            x, y = x.cuda(), y.cuda().float().unsqueeze(1)

        opt_fun.zero_grad()

        pred = model(x)
        loss = loss_fun(pred, y)

        loss.backward()
        opt_fun.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fun):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            if on_gpu:
                x, y = x.cuda(), y.cuda().float().unsqueeze(1)
            pred = model(x)
            test_loss += loss_fun(pred.float(), y.float()).item()
            # TODO 计算正确率

    test_loss /= num_batches
    correct /= size
    print(f"平均损失 : {test_loss}")


if __name__ == "__main__":
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, resnet18, loss_fn, optimizer)
        test_loop(val_dataloader, resnet18, loss_fn)
    print("Done!")
