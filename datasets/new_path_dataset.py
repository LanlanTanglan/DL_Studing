import os

from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

"""
输入x为[<begin>,a1,a2,a3,a4,a5]
输出y为[a1,a2,a3,a4,a5,<end>]
"""

import os

from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

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


class PathDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.path_img_path = os.listdir(os.path.join(root_dir, 'path_img'))
        self.split_data = []  # (img_id,[path_point,...])
        for img in self.path_img_path:
            idx = img.split('.')[0]
            with open(os.path.join(root_dir, 'path_info', idx + '.txt'), 'r') as f:
                lines = [(int(x), int(y)) for line in f for x, y in [line.strip().split(',')]]
            # 将x与y都变为裁切后的图片
            x = lines[:-1]
            y = lines[1:]
            padx = []
            pady = []
            for i in range(tgt_len - len(x)):
                padx += [(-1e9, -1e9)]
            for i in range(tgt_len - len(x)):
                pady += [(0, 0)]
            self.split_data += [(img, x + padx, y + pady)]

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        img, input_paths, output_paths = self.split_data[idx]
        img = Image.open(os.path.join(self.root_dir, 'path_img', img)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        input_patches = cut_patch(img, input_paths)
        # return img, torch.stack(input_patches), torch.stack(output_patches), torch.tensor(input_paths), torch.tensor(
        #     output_paths)

        return torch.stack(input_patches), torch.tensor(output_paths), img
