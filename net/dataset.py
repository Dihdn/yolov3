import torch
from torch import nn
from config import *
from utilis_2 import *
from yolov3 import *
from torch.utils.data import Dataset
from PIL import Image
import os


class Yolo_dataset(Dataset):
    def __init__(self):
        super().__init__()
        with open(img_name_path, 'r', encoding='utf8') as f:
            self.img_name = f.read()
        self.img_name = self.img_name.split('\n')

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_path = os.path.join(img_basic, self.img_name[item])  # 拼接成完整一张图片地址
        img = Image.open(img_path).convert('RGB')  # 打开图片
        img = make_416_img(img)  # 将图片填充并缩放为416*416大小

        label = get_info(self.img_name[item], [13, 13], anchors[0])
        return img, label


if __name__ == '__main__':
    dataset = Yolo_dataset()
    img = dataset[0]
