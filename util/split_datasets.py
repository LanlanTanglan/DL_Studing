# 将train数据按照6 2 2进行划分
import os
import shutil

# 以下三个必须和为10, 分别为train, val, test
ratio = [6, 2, 2]

if __name__ == "__main__":
    path = "..\\datasets\\apple_banana_datasets"
    train_path = os.path.join(path, "train")
    labels = [f.name for f in os.scandir(train_path) if f.is_dir()]
    if not os.path.exists(os.path.join(path, "val")):
        os.makedirs(os.path.join(path, "val"))
    if not os.path.exists(os.path.join(path, "test")):
        os.makedirs(os.path.join(path, "test"))
    if len(os.listdir(os.path.join(path, "val"))) == 0:
        for label in labels:
            # 创建文件夹
            if not os.path.exists(os.path.join(path, "val", label)):
                os.makedirs(os.path.join(path, "val", label))
            if not os.path.exists(os.path.join(path, "test", label)):
                os.makedirs(os.path.join(path, "test", label))
            img_paths = os.listdir(os.path.join(train_path, label))
            total_length = len(img_paths)
            part_lengths = [length * total_length // sum(ratio) for length in ratio]

            part2 = img_paths[part_lengths[0]:part_lengths[0] + part_lengths[1]]
            part3 = img_paths[-part_lengths[2]:]

            for p in part2:
                shutil.move(os.path.join(train_path, label, p), os.path.join(path, "val", label))
            for p in part3:
                shutil.move(os.path.join(train_path, label, p), os.path.join(path, "test", label))
