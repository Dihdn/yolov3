import math
import os
import torch
from PIL import Image
from config import *
import xml.etree.ElementTree as ET


# 将图片填充并缩放为416*416大小
def make_416_img(img):
    w, h = img.size
    new = max(w, h)
    img_new = Image.new('RGB', (new, new))
    img_new.paste(img, box=(0, 0))
    img_new = img_new.resize((416, 416))
    return img_new


# 制作标签
def get_info(img_name, w_and_h, anchors):  # img_name: 图像名字 w_and_h: 特征图的宽和高 anchors: 该特征图下的三个先验框
    img_name = img_name.split('.')[0]
    img_xml_path = os.path.join(xml_basic, f"{img_name}.xml")
    tree = ET.parse(img_xml_path)
    root = tree.getroot()  # 获取根节点
    size = root.find('size')  # 获取真实宽高
    w_ori = int(size.find('width').text)  # 获取原图像的宽
    h_ori = int(size.find('height').text)  # 获取原图像的高
    ori_max = max(w_ori, h_ori)  # 图像扩充之后的宽和高
    objects = list(root.findall('object'))  # 获取所有目标
    label = torch.zeros((3, w_and_h[0], w_and_h[1], (1+4+CLASS_SUM)))  # 为三个先验框绘制特征图网格存标签
    for i, anchor in enumerate(anchors):
        anchor_w, anchor_h = anchor[0], anchor[1]
        for object in objects:
            name = object.find('name').text
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)  # 真实框在原图像左上角x坐标
            ymin = int(bndbox.find('ymin').text)  # 真实框在原图像左上角y坐标
            xmax = int(bndbox.find('xmax').text)  # 真实框在原图像右下角x坐标
            ymax = int(bndbox.find('ymax').text)  # 真实框在原图像右下角y坐标
            x_center = (xmax - xmin) / 2  # 真实框在原图像中心坐标x
            y_center = (ymax - ymin) / 2  # 真实框在原图像中心坐标y
            anchor_xmin = x_center-anchor_w/2  # 先验框在原图像左上角x坐标
            anchor_xmax = x_center+anchor_w/2  # 先验框在原图像左上角y坐标
            anchor_ymin = y_center-anchor_h/2  # 先验框在原图像右下角x坐标
            anchor_ymax = y_center+anchor_h/2  # 先验框在原图像右下角y坐标
            w_obj = int(xmax - xmin)  # 真实框在原来图像的宽
            h_obj = int(ymax - ymin)   # 真实框在原来图像的宽
            w_416 = w_obj*(img_w / ori_max)  # 将真实框宽映射到416*416图像中
            h_416 = h_obj * (img_h / ori_max)  # 将真实框高映射到416*416图像中
            case_x = int(img_w/w_and_h[0])  # 416*416到特征图像宽的缩放因子
            case_y = int(img_w / w_and_h[1])  # 416*416到特征图像宽的缩放因子
            tx, x_index = math.modf(x_center/case_x)  # 真实框的中心坐标x映射到特征图中的小数部分和整数部分
            ty, y_index = math.modf(y_center/case_y)  # 真实框的中心坐标y映射到特征图中的小数部分和整数部分
            tw = math.log(w_416/anchor_w)
            th = math.log(h_416/anchor_h)
            one_hot = get_one_hot(CLASS_SUM, name)
            confidence = get_confidence([xmin, ymin, xmax, ymax],
                                        [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax], one_hot)
            # tx = torch.tensor(tx, dtype=torch.float32)
            # ty = torch.tensor(ty, dtype=torch.float32)
            # tw = torch.tensor(tw, dtype=torch.float32)
            # th = torch.tensor(th, dtype=torch.float32)
            label[i][int(x_index)][int(y_index)][:5] = torch.tensor([confidence, tx, ty, tw, th])
            label[i][int(x_index)][int(y_index)][5:] = one_hot
    return label


# 将类别转成one-hot
def get_one_hot(class_num, class_name, is_smoothing=True, epsilon=0.1):
    """

    :param class_num: 类别总数
    :param class_name: 类别名
    :param is_smoothing: 是否要平滑，默认是
    :param epsilon: 平滑因子
    :return: ont-hot类型
    """
    one_hot = torch.zeros(class_num)
    index = get_index(CLASS_NAMES, class_name)
    one_hot[index] = 1
    if is_smoothing:
        one_hot = label_smoothing(one_hot, class_num, epsilon)
    return one_hot


# ont-hot平滑
def label_smoothing(label, class_num, epsilon=0.1):
    """

    :param label: one-hot类型
    :param class_num: 类别总数
    :param epsilon: 平滑因子
    :return: 平滑后的one-hot
    """
    onr_hot = torch.ones(label.shape)
    ont_hot = (epsilon/class_num)*onr_hot+(1-epsilon)*label
    return ont_hot


# 获取相关分类的下标
def get_index(class_list, class_name):
    for i, name in enumerate(class_list):
        if name == class_name:
            return i


# 计算IOU
def get_iou(box1, box2):
    """

    :param box1: 第一个框的左上角和右下角坐标
    :param box2: 第二个框的左上角和右下角坐标
    :return: iou
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    iou = ((x2-x1)*(y2-y1))/((box1[2]-box1[0])*(box1[3]-box1[1])+(box2[2]-box2[0])*(box2[3]-box2[1]))
    if iou < 0:
        return 0
    return iou


# 计算confidence
def get_confidence(box1, box2, one_hot):
    """

    :param box1: 第一个框的左上角和右下角坐标
    :param box2: 第二个框的左上角和右下角坐标
    :param one_hot: one-hot类型
    :return: confidence
    """
    iou = get_iou(box1, box2)
    precision, _ = one_hot.max(dim=0)
    return iou*precision


if __name__ == '__main__':
    label = get_info('2007_000027', [13, 13], anchors[0])
    print(label[0][2][3])
    print(label[1][2][3])
    print(label[2][2][3])
