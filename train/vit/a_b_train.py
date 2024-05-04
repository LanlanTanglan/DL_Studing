import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from DL_Studing.datasets.a_b_dataset import ABDataset
from DL_Studing.models.vit import vit_base_patch16_224
from torch.utils.tensorboard import SummaryWriter

"""
超参数
"""
learning_rate = 1e-4
batch_size = 64
epochs = 100
on_gpu = True

# 启用tensorboard
tb_writer = SummaryWriter(log_dir="..\\..\\runs\\logs")

# 变换
default_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 载入数据集
train_data = ABDataset("..\\..\\data\\apple_banana_datasets\\train", default_transform)
val_data = ABDataset("..\\..\\data\\apple_banana_datasets\\val", default_transform)
test_data = ABDataset("..\\..\\data\\apple_banana_datasets\\test", default_transform)

# 分类数量标签
labels = ['apple', 'banana']

# 载入数据集加载器
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

vit_b_16 = vit_base_patch16_224(num_classes=len(labels) - 1).cuda()
loss_fn = nn.BCELoss()
vit_b_16_optim = torch.optim.SGD(vit_b_16.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fun, opt_fun, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        if on_gpu:
            x, y = x.cuda(), y.cuda().float().unsqueeze(1)

        opt_fun.zero_grad()
        pred = model(x)
        pred = nn.Sigmoid()(pred)
        loss = loss_fun(pred, y)
        train_loss += loss.item()
        loss.backward()
        opt_fun.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss = train_loss / num_batches
    # 绘制多条曲线
    tb_writer.add_scalar('Train/Loss', train_loss, epoch)


def test_loop(dataloader, model, loss_fun, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            if on_gpu:
                x, y = x.cuda(), y.cuda().float().unsqueeze(1)
            pred = model(x)
            pred = nn.Sigmoid()(pred)
            test_loss += loss_fun(pred.float(), y.float()).item()
            # TODO 计算正确率

    test_loss /= num_batches
    correct /= size
    print(f"平均损失 : {test_loss}")
    # tb_writer.add_scalar("loss/val", test_loss, epoch)
    tb_writer.add_scalar('Test/Loss', test_loss, epoch)


if __name__ == "__main__":
    # TODO 绘制模型
    init_img = torch.zeros((1, 3, 224, 224), device="cuda")
    tb_writer.add_graph(vit_b_16, init_img)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, vit_b_16, loss_fn, vit_b_16_optim, epoch)
        test_loop(val_dataloader, vit_b_16, loss_fn, epoch)
    print("Done!")
    tb_writer.close()
