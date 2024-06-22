import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        self.full_one = torch.full((1, config.n_t, config.n_embd), 1)
        self.w_u = nn.Parameter(torch.full((config.n_t, config.n_embd), 1.0))
        self.pos_w = torch.zeros(config.n_t, config.n_attn)
        for i in range(config.n_t - 1):
            self.pos_w[i] = 2 - config.n_t + i
        self.new_pos_w = torch.zeros(config.n_t - 1, config.n_t, config.n_attn)
        for i in range(config.n_t - 1):
            self.new_pos_w[i] = torch.cat((self.pos_w[i:], torch.zeros(i, config.n_attn)))
        self.new_pos_one = torch.zeros(config.n_t - 1, config.n_t, config.n_attn)
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
        self.full_one = torch.full((1, config.n_t, config.n_embd), 1)
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
        self.layerNorm = nn.LayerNorm(config.n_embd)
        self.TimeMixing = TimeMixing(config)
        self.ChannelMixing = ChannelMixing(config)

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


class Config:
    def __init__(self, n_t, n_embd, n_attn, layer):
        self.n_embd = n_embd
        self.n_attn = n_attn
        self.n_t = n_t
        self.layer = layer


if __name__ == '__main__':
    cf = Config(12, 2, 12, 8)
    timem = RWKVv1(cf)
    x = torch.randn(1, cf.n_t, cf.n_embd)
    timem(x)
