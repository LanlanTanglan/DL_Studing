import os

from PIL import Image
from torch.utils.data import Dataset

"""
数据集的文件夹格式
--train
    --label1
    --label2
"""


class ABDataset(Dataset):
    """
    苹果香蕉分类Dataset
    """

    def __init__(self, img_dir, transform=None, target_transform=None):
        # 根据img_dir的类别文件夹数量，构建标签列表labels
        self.labels = [f.name for f in os.scandir(img_dir) if f.is_dir()]
        # 构建图片路径对应类别的列表，如(img1_path, img1_label)
        # TODO 这里的标签我用的是字符串，改成数字也可以
        self.imgs = []
        for label in self.labels:
            # 获取这个标签下面的所有文件
            for img_file in os.listdir(os.path.join(img_dir, label)):
                self.imgs.append((os.path.join(img_dir, label, img_file), label))
        # 图像变换器
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # 返回数据集长度
        return len(self.imgs)

    # 返回的一定是一个img以及一个标签
    def __getitem__(self, idx):
        img_file_path, label = self.imgs[idx]
        img = Image.open(img_file_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img.resize((224, 224), Image.NEAREST))
        return img, self.labels.index(label)
