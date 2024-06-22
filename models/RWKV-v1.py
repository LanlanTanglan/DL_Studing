import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from DL_Studing.datasets.new_path_dataset import PathDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Transformer Parameters
d_model = 27  # Embedding Size
d_ff = d_model * 4  # FeedForward dimension
d_k = d_v = 9  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 3  # number of heads in Multi-Head Attention
tgt_len = 128
batch_size = 4
img_w, img_h = 32, 32


def cut_patch(img, pos, patch_size=3):
    patches = []
    for coord in pos:
        x, y = coord[0], coord[1]
        # 计算图像块的位置
        if x == -1e9 and y == -1e9:
            patches.append(torch.full((3, patch_size, patch_size), -1e9))
            continue
        x_start = max(x - patch_size // 2, 0)
        x_end = min(x + patch_size // 2 + 1, img.size(1))
        y_start = max(y - patch_size // 2, 0)
        y_end = min(y + patch_size // 2 + 1, img.size(2))

        # 计算边界的补全量
        left_pad = patch_size // 2 if y == 0 else 0
        right_pad = patch_size // 2 if y == img.size(1) - 1 else 0
        bottom_pad = patch_size // 2 if x == img.size(2) - 1 else 0
        top_pad = patch_size // 2 if x == 0 else 0

        # 使用pad函数进行补全
        patch = F.pad(img[:, x_start:x_end, y_start:y_end], (left_pad, right_pad, top_pad, bottom_pad))

        # 将提取的图像块添加到列表中
        patches.append(patch)
    return patches


class InputEmbedding(nn.Module):
    def __init__(self):
        super(InputEmbedding, self).__init__()

    def forward(self, x):
        return x


class TimeMixing(nn.Module):
    def __init__(self, config):
        super(TimeMixing, self).__init__()
        self.W_R = nn.Linear(config.n_embd, config.n_attn)
        self.W_K = nn.Linear(config.n_embd, config.n_attn)
        self.W_V = nn.Linear(config.n_embd, config.n_attn)
        self.mu_R = nn.Parameter(torch.full((1, config.n_t, config.n_embd), 0.5))
        self.mu_K = nn.Parameter(torch.full((1, config.n_t, config.n_embd), 0.5))
        self.mu_V = nn.Parameter(torch.full((1, config.n_t, config.n_embd), 0.5))
        self.sigmoid = nn.Sigmoid()
        self.full_one = torch.full((1, config.n_t, config.n_embd), 1).cuda()
        self.w_u = nn.Parameter(torch.full((config.n_t, config.n_embd), 1.0))
        self.pos_w = torch.zeros(config.n_t, config.n_attn).cuda()
        for i in range(config.n_t - 1):
            self.pos_w[i] = 2 - config.n_t + i
        self.new_pos_w = torch.zeros(config.n_t - 1, config.n_t, config.n_attn).cuda()
        for i in range(config.n_t - 1):
            self.new_pos_w[i] = torch.cat((self.pos_w[i:], torch.zeros(i, config.n_attn).cuda()))
        self.new_pos_one = torch.zeros(config.n_t - 1, config.n_t, config.n_attn).cuda()
        oee = torch.ones(config.n_t, config.n_attn)
        for i in range(config.n_t - 1):
            self.new_pos_one[i] = torch.cat((oee[i:], torch.zeros(i, config.n_attn)))
        self.W_O = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        zero_part = torch.zeros(B, 1, C).to(x.device)
        x_copy = x.clone()
        x_copy = torch.cat((x_copy[:, 1:, :], zero_part), dim=1)
        mu_r_e = self.mu_R.expand(B, -1, -1)
        mu_k_e = self.mu_K.expand(B, -1, -1)
        mu_v_e = self.mu_V.expand(B, -1, -1)
        full_one = self.full_one.expand(B, -1, -1)
        R = self.W_R(mu_r_e * x + (full_one - mu_r_e) * x_copy)
        R_E = R.unsqueeze(1).expand(-1, T - 1, -1, -1)
        K = self.W_K(mu_k_e * x + (full_one - mu_k_e) * x_copy)
        K_E = K.unsqueeze(1).expand(-1, T - 1, -1, -1)
        V = self.W_V(mu_v_e * x + (full_one - mu_v_e) * x_copy)
        V_E = V.unsqueeze(1).expand(-1, T - 1, -1, -1)
        new_pos_one_e = self.new_pos_one.expand(B, -1, -1, -1)
        new_pos_w_e = self.new_pos_w.expand(B, -1, -1, -1)
        top = torch.exp(new_pos_w_e + K_E * new_pos_one_e)
        wkv = torch.sum(top * V_E, dim=1)
        rwkv = self.W_O(torch.sigmoid(R) * wkv / torch.sum(top, dim=1))
        return rwkv  # rwkv : [B, n_t, n_attn]


class ChannelMixing(nn.Module):
    def __init__(self, config):
        super(ChannelMixing, self).__init__()
        self.W_R = nn.Linear(config.n_embd, config.n_attn)
        self.W_K = nn.Linear(config.n_embd, config.n_attn)
        self.W_V = nn.Linear(config.n_embd, config.n_attn)
        self.V_V = nn.Linear(config.n_attn, config.n_attn)
        self.W_O = nn.Linear(config.n_attn, config.n_embd)
        self.full_one = torch.full((1, config.n_t, config.n_embd), 1).cuda()
        self.mu_R = nn.Parameter(torch.full((1, config.n_t, config.n_embd), 0.5))
        self.mu_K = nn.Parameter(torch.full((1, config.n_t, config.n_embd), 0.5))
        self.mu_V = nn.Parameter(torch.full((1, config.n_t, config.n_embd), 0.5))
        self.relu = nn.ReLU()

    def forward(self, x):
        B, T, C = x.size()
        zero_part = torch.zeros(B, 1, C).to(x.device)
        x_copy = x.clone()
        x_copy = torch.cat((x_copy[:, 1:, :], zero_part), dim=1)
        mu_r_e = self.mu_R.expand(B, -1, -1)
        mu_k_e = self.mu_K.expand(B, -1, -1)
        mu_v_e = self.mu_V.expand(B, -1, -1)
        full_one = self.full_one.expand(B, -1, -1)
        R = self.W_R(mu_r_e * x + (full_one - mu_r_e) * x_copy)
        K = self.W_K(mu_k_e * x + (full_one - mu_k_e) * x_copy)
        V = self.W_V(mu_v_e * x + (full_one - mu_v_e) * x_copy)
        rwkv = self.W_O(torch.sigmoid(R) * torch.pow(self.V_V(self.relu(K)), 2))
        return rwkv


class RwkvBlock(nn.Module):
    def __init__(self, config, layer_id):
        super(RwkvBlock, self).__init__()
        self.layer_id = layer_id
        self.layerNorm = nn.LayerNorm(config.n_embd).cuda()
        self.TimeMixing = TimeMixing(config).cuda()
        self.ChannelMixing = ChannelMixing(config).cuda()

    def forward(self, x):
        res_x = x
        x = self.layerNorm(res_x)
        x = self.TimeMixing(x)
        x = x + res_x
        res_x = x
        x = self.layerNorm(res_x)
        x = self.ChannelMixing(x)
        x = x + res_x
        return x


class RwkvDecoder(nn.Module):
    def __init__(self, config):
        super(RwkvDecoder, self).__init__()

    def forward(self, x):
        return x


class PathRwkvEncode(nn.Module):
    def __init__(self, config, dim=3, patch_size=3, ):
        super(PathRwkvEncode, self).__init__()
        self.config = config
        self.patch_size = patch_size
        self.max_len = config.n_t
        self.dim = dim
        self.pad = torch.zeros(1, config.n_t, config.n_embd)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=patch_size * patch_size, stride=10)

    # img: 1 * 3 * 32 * 32
    # x : 1 * len * 3 * 3 * 3
    def forward(self, x, img):
        # x: B * len * 27
        # img: B * ch * w * h
        x = x.reshape(-1, x.size(1), self.config.n_embd)
        img = self.conv1(img)
        img = img.expand(3, -1, -1, -1, -1)
        img = img.permute(1, 2, 3, 4, 0)
        img = img.reshape(-1, img.size(1), self.config.n_embd)  # img : batch * len * 27
        x = img + x
        # cls_token = nn.Parameter(torch.full((x.size(0), 1, d_model), 0.0)).cuda()
        # x = torch.cat((x, cls_token), dim=1)

        return x


class PathRwkvDecode(nn.Module):
    def __init__(self, config, patch_size=3):
        super(PathRwkvDecode, self).__init__()
        self.linear = nn.Linear(config.n_embd, 2)
        self.linear1 = nn.Linear(in_features=config.n_embd, out_features=config.n_embd * 4, bias=False)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(in_features=config.n_embd * 4, out_features=2, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        return x


class RWKVv1(nn.Module):
    def __init__(self, config):
        super(RWKVv1, self).__init__()
        self.layerNorm = nn.LayerNorm(config.n_embd)
        self.blocks = [RwkvBlock(config, i) for i in range(config.layer)]
        self.inputEmbedding = InputEmbedding()
        self.rwkvDecoder = RwkvDecoder(config)

    def forward(self, x):
        x = self.inputEmbedding(x)
        x = self.layerNorm(x)
        for block in self.blocks:
            x = block(x)
        # 各种各样的头
        x = self.rwkvDecoder(x)
        return x


class PathRwkv(nn.Module):
    def __init__(self, config):
        super(PathRwkv, self).__init__()
        self.layerNorm = nn.LayerNorm(config.n_embd)
        self.blocks = [RwkvBlock(config, i) for i in range(config.layer)]
        self.path_encoder = PathRwkvEncode(config)
        self.path_decoder = PathRwkvDecode(config)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, img):
        x = self.path_encoder(x, img)
        x = self.layerNorm(x)
        for block in self.blocks:
            x = self.dropout(x)
            x = block(x)
        x = self.path_decoder(x)
        return x


class Config:
    def __init__(self, n_t, n_embd, n_attn, layer):
        self.n_embd = n_embd
        self.n_attn = n_attn
        self.n_t = n_t
        self.layer = layer


if __name__ == '__main__':
    cf = Config(128, 27, 54, 6)
    pt_model = PathRwkv(cf).cuda()
    loss_fun = nn.MSELoss()
    # optimizer = optim.SGD(pt_model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = torch.optim.Adam(pt_model.parameters(), lr=1e-4)
    on_gpu = True

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = PathDataset("D:\\DeepLearning\\PersonalStudy\\DL_Studing\\data\\pixel_path\\train",
                             transform=transform)
    tb_writer = SummaryWriter(log_dir="D:\\DeepLearning\\PersonalStudy\\DL_Studing\\runs\\path_rwkv")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(12000):
        train_loss = 0
        pt_model.train()
        for batch, (x, y_pos, img) in enumerate(train_dataloader):
            if on_gpu:
                x, y_pos, img = x.cuda().float(), y_pos.cuda().float(), img.cuda().float()

            optimizer.zero_grad()
            pred = pt_model(x, img)  # pred : B * len * 2
            zero_tensor = torch.zeros(batch_size, 1, 2).cuda()
            x_pos = torch.cat((zero_tensor, y_pos), dim=1)
            # 去掉最右侧 batch*1*2
            x_pos = x_pos[:, :-1, :]
            # 给x_pos加上偏置
            y_pos_pred = x_pos + pred
            loss = loss_fun(y_pos_pred, y_pos)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"epoch {epoch}, train loss: {train_loss / len(train_data)}")
        tb_writer.add_scalar('train/Loss', train_loss / len(train_data), epoch)
    torch.save(pt_model.state_dict(),
               'D:\\DeepLearning\\PersonalStudy\\DL_Studing\\weights\\path_rwkv\\path_rwkv_model1.pt')

    test_data = PathDataset("D:\\DeepLearning\\PersonalStudy\\DL_Studing\\data\\pixel_path\\test",
                            transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    for batch, (x, y_pos, img) in enumerate(test_dataloader):
        pt_model.eval()
        if on_gpu:
            x, y_pos, img = x.cuda(), y_pos.cuda().float(), img.cuda().float()
        stop = False
        cnt = 0
        img_squeezed = torch.squeeze(img, dim=0).cuda()
        seq = [(0, 0)]
        while not stop and cnt < tgt_len:
            padx = []
            for i in range(tgt_len - len(seq)):
                padx += [(-1e9, -1e9)]
            x_seq = seq + padx
            x_inputs_patches = [x.to('cuda') for x in cut_patch(img_squeezed, x_seq)]
            x_inputs = torch.stack(x_inputs_patches).unsqueeze(0)
            pred = pt_model(x_inputs, img)
            offset = pred.squeeze(0)[cnt].round().squeeze(0)
            next_pos_x = int((seq[cnt][0] + offset[0]).item())
            next_pos_y = int((seq[cnt][1] + offset[1]).item())
            seq += [(next_pos_x, next_pos_y)]
            print(
                f"before : {seq[cnt]}"
                f", pred : {(next_pos_x, next_pos_y)}"
                f", target : {y_pos.squeeze(0)[cnt]}")
            cnt += 1
            if next_pos_x < 0 or next_pos_y < 0:
                stop = True
            if next_pos_x >= 32 or next_pos_y >= 32:
                stop = True
            continue
