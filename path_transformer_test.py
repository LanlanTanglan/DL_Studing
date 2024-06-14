import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DL_Studing.datasets.path_dataset import PathDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from models.new_path_transformer import PathTransformer
import os

from PIL import Image
from torch.utils.data import Dataset
import torch

transform = transforms.Compose([transforms.ToTensor()])
on_gpu = True if torch.cuda.is_available() else False
tgt_len = 128


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


if __name__ == "__main__":
    state_dict = torch.load("weights/path_transformer/path_transformer_model1.pt")
    pt_model = PathTransformer().cuda()
    pt_model.load_state_dict(state_dict)
    path_result = [(0, 0)]
    img = Image.open("data/pixel_path/test/path_img/1.png").convert('RGB')
    if transform:
        img = transform(img)
    pt_model.eval()
    stop = False
    cnt = 0
    img_unsqueezed = torch.unsqueeze(img, dim=0).cuda()
    seq = [(0, 0)]
    while not stop and cnt < tgt_len:
        padx = []
        for i in range(tgt_len - len(seq)):
            padx += [(-1e9, -1e9)]
        x_seq = seq + padx
        x_inputs_patches = [x.to('cuda') for x in cut_patch(img, x_seq)]
        x_inputs = torch.stack(x_inputs_patches).unsqueeze(0)
        pred = pt_model(x_inputs, img_unsqueezed)
        offset = pred.squeeze(0)[cnt].round().squeeze(0)
        next_pos_x = int((seq[cnt][0] + offset[0]).item())
        next_pos_y = int((seq[cnt][1] + offset[1]).item())
        seq += [(next_pos_x, next_pos_y)]
        print(
            f"before : {seq[cnt]}"
            f", pred : {(next_pos_x, next_pos_y)}")
        cnt += 1
        if next_pos_x < 0 or next_pos_y < 0:
            stop = True
        if next_pos_x >= 32 or next_pos_y >= 32:
            stop = True
        continue

    # while path_result[-1][0] != 31 or path_result[-1][1] != 31:
    #     pad = []
    #     for k in range(tgt_len - len(path_result)):
    #         pad += [(0, 0)]
    #     pad_path_result = pad + path_result
    #     x = torch.stack(cut_patch(img, pad_path_result)).cuda().unsqueeze(0)
    #     x1 = torch.tensor(pad_path_result).cuda().unsqueeze(0)
    #     input_img = img.cuda().float().unsqueeze(0)
    #     pt_model.eval()
    #     pred = pt_model(x, x1, input_img)
    #     pred_token = (pred[:, -1:pred.size(1), :] @ (1 * torch.eye(2).cuda())) + x1[:, -1:x1.size(1), :].cuda()
    #     next_pos = torch.round(pred_token).int().squeeze(0).squeeze(0)
    #     path_result += [
    #         (next_pos[0].item() if next_pos[0].item() != 32 else 31, next_pos[1].item()) if next_pos[1] != 32 else 31]
    #     #     if on_gpu:
    #     #         x, y = x.cuda(), y.cuda().float()
    #     #     pred = pt_model(x, x1)
    #     #     pred_token = (pred[:, -1:pred.size(1), :] @ (1 * torch.eye(2).cuda())) + x1[:, -1:x1.size(1), :].cuda()
    #     #     print(
    #     #         f"pred : {pred_token}"
    #     #         f", before : {x1[:, -1:x1.size(1), :].cuda()}"
    #     #         f", target : {y}")
    #
    #     continue
    #
    # # test_data = PathDataset("D:\\DeepLearning\\PersonalStudy\\DL_Studing\\data\\pixel_path\\test",
    # #                         transform=transform)
    # # test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    # # for batch, (x, x1, y) in enumerate(test_dataloader):
    # #     pt_model.eval()
    # #     if on_gpu:
    # #         x, y = x.cuda(), y.cuda().float()
    # #     pred = pt_model(x, x1)
    # #     pred_token = (pred[:, -1:pred.size(1), :] @ (1 * torch.eye(2).cuda())) + x1[:, -1:x1.size(1), :].cuda()
    # #     print(
    # #         f"pred : {pred_token}"
    # #         f", before : {x1[:, -1:x1.size(1), :].cuda()}"
    # #         f", target : {y}")
