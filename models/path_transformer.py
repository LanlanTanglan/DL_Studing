import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DL_Studing.datasets.path_dataset import PathDataset
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
batch_size = 256
img_w, img_h = 32, 32


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
    def forward(self, x, x1, img):
        # x: B * len * 27
        # x1: B * len * 2
        # img: B * ch * w * h
        x = x.reshape(-1, x.size(1), d_model)
        # img = img.reshape(-1, img.size(1), img_w * img_h)
        # img = img.permute(0, 2, 3, 1)
        img = self.conv1(img)
        img = img.expand(3, -1, -1, -1, -1)
        img = img.permute(1, 2, 3, 4, 0)
        img = img.reshape(-1, img.size(1), d_model)
        x = img + x
        cls_token = nn.Parameter(torch.full((x.size(0), 1, d_model), 0.0)).cuda()
        x = torch.cat((x, cls_token), dim=1)

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

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
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

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
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
        for layer in self.layers:
            dec_inputs, dec_self_attn = layer(dec_inputs)
            dec_self_attns.append(dec_self_attn)
        return dec_inputs, dec_self_attns


class PathTransformer(nn.Module):
    def __init__(self):
        super(PathTransformer, self).__init__()
        self.dire_ecn = DirectionEncode()
        self.decoder = Decoder()
        self.dire_dec = DirectionDecode()

    def forward(self, dec_inputs, x1, img):
        dec_outputs = self.dire_ecn(dec_inputs, x1, img)  # [batch_size, tgt_len, d_model]
        dec_outputs, dec_self_attns = self.decoder(dec_outputs)
        dec_logits = self.dire_dec(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # return dec_logits.view(-1, dec_logits.size(-1))
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
    for epoch in range(1200):
        train_loss = 0
        pt_model.train()
        for batch, (x, x1, y, img) in enumerate(train_dataloader):
            if on_gpu:
                x, y, img = x.cuda().float(), y.cuda().float(), img.cuda().float()

            optimizer.zero_grad()
            pred = pt_model(x, x1, img)  # pred : B * len * 2
            pred_token = (pred[:, -1:pred.size(1), :] @ (1 * torch.eye(2).cuda())) + x1[:, -1:x1.size(1), :].cuda()
            pred_token = pred_token.squeeze(1)
            loss = loss_fun(
                pred_token,
                y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"epoch {epoch}, train loss: {train_loss / len(train_data)}")
    torch.save(pt_model.state_dict(),
               'D:\\DeepLearning\\PersonalStudy\\DL_Studing\\weights\\path_transformer\\model4.pt')

    test_data = PathDataset("D:\\DeepLearning\\PersonalStudy\\DL_Studing\\data\\pixel_path\\test",
                            transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    for batch, (x, x1, y, img) in enumerate(test_dataloader):
        pt_model.eval()
        if on_gpu:
            x, y, img = x.cuda(), y.cuda().float(), img.cuda().float()
        pred = pt_model(x, x1, img)
        pred_token = (pred[:, -1:pred.size(1), :] @ (1 * torch.eye(2).cuda())) + x1[:, -1:x1.size(1), :].cuda()
        print(
            f"pred : {pred_token}"
            f", before : {x1[:, -1:x1.size(1), :].cuda()}"
            f", target : {y}")
