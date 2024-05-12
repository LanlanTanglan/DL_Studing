import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from random import randint


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1, 1000)
        self.l2 = nn.Linear(1000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.relu = nn.ReLU()
        return

    def forward(self, x):
        x = self.l1(x)
        # x = self.relu(x)
        x = self.l2(x)
        # x = self.relu(x)
        x = self.l3(x)
        return x

    pass


lr = 1e-3
my_model = Model().cuda()
my_loss = nn.MSELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=lr)
tb_writer = SummaryWriter(log_dir="runs\\logs\\test")


def generate_data_set(size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    data = []
    for i in range(size):
        n = randint(-100, 100)
        data.append((torch.tensor([n], dtype=torch.float),
                     torch.tensor([n + 114514], dtype=torch.float)))
    return data


def test(model: torch.nn.Module, epoch):
    right = 0
    test_set = generate_data_set(114)
    with torch.no_grad():
        for x, y in test_set:
            z = model(x)
            print(f"input x: {x.item()},des_result: {y.item()},prediction: {z.item()},away: {abs(z - y).item()}")
            if abs(z[0] - y[0]) < 1:
                right += 1
    print(f'Accurate rate:{right / 114}')
    tb_writer.add_scalar('Accurate rate', right / 114, epoch)


if __name__ == '__main__':
    torch.set_default_device('cuda')

    for epoch in range(114):
        my_model.train()
        for x, y in generate_data_set(1145):
            optimizer.zero_grad()
            pred = my_model(x)
            loss = my_loss(pred, y)
            loss.backward()
            optimizer.step()
        test(my_model, epoch)
