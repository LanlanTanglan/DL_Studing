import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.cat_dog_dataset import CatDogDataset
from models.resnet import ResNet18, ResNet34, NoIdentityNet18, NoIdentityNet34
from torch.utils.tensorboard import SummaryWriter

"""
超参数
"""
learning_rate = 1e-3
train_batch_size = 64
val_batch_size = 64
epochs = 100
on_gpu = True

# 启用tensorboard
tb_writer = SummaryWriter(log_dir="runs/logs")

# 变换
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 载入数据集
train_data = CatDogDataset("data\\cat_dog\\train", default_transform)
test_data = CatDogDataset("data\\cat_dog\\test", default_transform)

# 分类数量标签
labels = ['cats', 'dogs']

# 载入数据集加载器
train_dataloader = DataLoader(train_data, batch_size=train_batch_size)
test_dataloader = DataLoader(test_data, batch_size=val_batch_size)

# 载入模型, 这里我们载入的是不同层数，是否带残差的四个模型
resnet18 = ResNet18(num_classes=len(labels) - 1).cuda()
resnet34 = ResNet34(num_classes=len(labels) - 1).cuda()
no_identity_net18 = NoIdentityNet18(num_classes=len(labels) - 1).cuda()
no_identity_net34 = NoIdentityNet34(num_classes=len(labels) - 1).cuda()

# 损失函数 随便选个先试试，因为是二分类任务
resnet18_loss_fn = nn.BCELoss()
resnet34_loss_fn = nn.BCELoss()
no_identity_net18_loss_fn = nn.BCELoss()
no_identity_net34_loss_fn = nn.BCELoss()
# 优化器 随便选个先试试
optimizer18 = torch.optim.SGD(resnet18.parameters(), lr=learning_rate)
optimizer34 = torch.optim.SGD(resnet34.parameters(), lr=learning_rate)
ni_optimizer18 = torch.optim.SGD(no_identity_net18.parameters(), lr=learning_rate)
ni_optimizer34 = torch.optim.SGD(no_identity_net34.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fun, opt_fun, epoch, tag=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        if on_gpu:
            x, y = x.cuda(), y.cuda().float().unsqueeze(1)

        opt_fun.zero_grad()
        pred = model(x)
        loss = loss_fun(pred, y)
        train_loss += loss.item()
        loss.backward()
        opt_fun.step()
        # if batch % 20 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(x)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss = train_loss / num_batches
    # 绘制多条曲线
    if tag == 1:
        tb_writer.add_scalars('resnet/train/loss', {'resnet18': train_loss}, epoch)
    elif tag == 2:
        tb_writer.add_scalars('resnet/train/loss', {'resnet34': train_loss}, epoch)
    elif tag == 3:
        tb_writer.add_scalars('ni_net/train/loss', {'ni_net18': train_loss}, epoch)
    elif tag == 4:
        tb_writer.add_scalars('ni_net/train/loss', {'ni_net34': train_loss}, epoch)


def test_loop(dataloader, model, loss_fun, epoch, tag=0):
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
    # tb_writer.add_scalar("loss/val", test_loss, epoch)
    if tag == 1:
        tb_writer.add_scalars('resnet/val/loss', {'resnet18': test_loss}, epoch)
    elif tag == 2:
        tb_writer.add_scalars('resnet/val/loss', {'resnet34': test_loss}, epoch)
    elif tag == 3:
        tb_writer.add_scalars('ni_net/val/loss', {'ni_net18': test_loss}, epoch)
    elif tag == 4:
        tb_writer.add_scalars('ni_net/val/loss', {'ni_net34': test_loss}, epoch)


if __name__ == "__main__":
    # TODO 绘制模型
    # init_img = torch.zeros((1, 3, 224, 224), device="cuda")
    # tb_writer.add_graph(resnet18, init_img)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, resnet18, resnet18_loss_fn, optimizer18, epoch, tag=1)
        test_loop(test_dataloader, resnet18, resnet18_loss_fn, epoch, tag=1)

        train_loop(train_dataloader, resnet34, resnet34_loss_fn, optimizer34, epoch, tag=2)
        test_loop(test_dataloader, resnet34, resnet34_loss_fn, epoch, tag=2)

        train_loop(train_dataloader, no_identity_net18, no_identity_net18_loss_fn, ni_optimizer18, epoch, tag=3)
        test_loop(test_dataloader, no_identity_net18, no_identity_net18_loss_fn, epoch, tag=3)

        train_loop(train_dataloader, no_identity_net34, no_identity_net34_loss_fn, ni_optimizer34, epoch, tag=4)
        test_loop(test_dataloader, no_identity_net34, no_identity_net34_loss_fn, epoch, tag=4)
    print("Done!")
    tb_writer.close()
