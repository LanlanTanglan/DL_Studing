import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DL_Studing.datasets.new_path_dataset import PathDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

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


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size(0), seq_q.size(1)
    batch_size, len_k = seq_k.size(), seq_k.size(1)
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class DirectionEncode(nn.Module):
    def __init__(self, dim=3, patch_size=3, max_len=tgt_len):
        super(DirectionEncode, self).__init__()
        self.patch_size = patch_size
        self.max_len = tgt_len
        self.dim = dim
        self.pad = torch.zeros(1, tgt_len, d_model)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=patch_size * patch_size, stride=10)

    # img: 1 * 3 * 32 * 32
    # x : 1 * len * 3 * 3 * 3
    def forward(self, x, img):
        # x: B * len * 27
        # img: B * ch * w * h
        x = x.reshape(-1, x.size(1), d_model)
        img = self.conv1(img)
        img = img.expand(3, -1, -1, -1, -1)
        img = img.permute(1, 2, 3, 4, 0)
        img = img.reshape(-1, img.size(1), d_model)  # img : batch * len * 27
        x = img + x
        # cls_token = nn.Parameter(torch.full((x.size(0), 1, d_model), 0.0)).cuda()
        # x = torch.cat((x, cls_token), dim=1)

        return x


class DirectionDecode(nn.Module):
    def __init__(self, patch_size=3):
        super(DirectionDecode, self).__init__()
        self.linear = nn.Linear(d_model, 2)
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_model * 4, bias=False)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_model * 4, out_features=2, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        # [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)\
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        output = self.dropout(output)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_self_attns, dec_enc_attns = [], []
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt(dec_self_attn_subsequence_mask, 0).cuda()  # [batch_size, tgt_len, tgt_len]
        for layer in self.layers:
            dec_inputs, dec_self_attn = layer(dec_inputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_inputs, dec_self_attns


class PathTransformer(nn.Module):
    def __init__(self):
        super(PathTransformer, self).__init__()
        self.dire_ecn = DirectionEncode()
        self.decoder = Decoder()
        self.dire_dec = DirectionDecode()

    def forward(self, dec_inputs, img):
        dec_outputs = self.dire_ecn(dec_inputs, img)  # [batch_size, tgt_len, d_model]
        dec_outputs, dec_self_attns = self.decoder(dec_outputs)
        dec_logits = self.dire_dec(dec_outputs)  # dec_logits: [batch_size, tgt_len, 2]
        return dec_logits


if __name__ == '__main__':
    pt_model = PathTransformer().cuda()
    loss_fun = nn.MSELoss()
    # optimizer = optim.SGD(pt_model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = torch.optim.Adam(pt_model.parameters(), lr=1e-4)
    on_gpu = True

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = PathDataset("D:\\DeepLearning\\PersonalStudy\\DL_Studing\\data\\pixel_path\\train",
                             transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(10000):
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
    # torch.save(pt_model.state_dict(),
    #            'D:\\DeepLearning\\PersonalStudy\\DL_Studing\\weights\\path_transformer\\model4.pt')

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
