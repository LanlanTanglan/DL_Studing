import os

if __name__ == "__main__":
    path = "datasets\\apple_banana_datasets\\train"
    labels = [f.name for f in os.scandir(path) if f.is_dir()]
    imgs = []
    # test
    for label in labels:
        # 获取这个标签下面的所有文件
        for img_file in os.listdir(os.path.join(path, label)):
            imgs.append((os.path.join(path, label, img_file), label))
    print(imgs)
