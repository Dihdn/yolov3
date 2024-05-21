residual_num = [1, 2, 8, 8, 4]  # darknet每个残差块的数量  # darknet.py
set_input = [128, 256, 512, 1024]  # 三个Convolutional set层的输入  # yolov3.py

CLASS_SUM = 20  # 总共类别数  # yolov3.py

CLASS_NAMES = ['person', 'bird', 'cat', 'cow',' dog', 'horse', 'sheep', 'aeroplane',
               'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
               ' dining table', 'potted plant', 'sofa', 'tv/monitor']

anchors = [[[116, 90], [158, 198], [373, 326]],  # 13*13特征图的先验框
           [[30, 61], [62, 45], [59, 119]],      # 26*26特征图的先验框
           [[10, 13], [16, 30], [33, 23]]]       # 52*52特征图的先验框

img_h = 416  # 输入图像的高
img_w = 416  # 输入图像的宽

xml_basic = 'E:\\object\\VOC2012\\Annotations'  # 图片标签的根目录地址
img_basic = 'E:\\object\\VOC2012\\JPEGImages'  # 图片根目录地址
img_name_path = 'E:\\object\\VOC2012\\img_name_path.txt'  # 图片名称地址文件路径

epochs = 10  # 训练轮数
LR = 0.02  # 学习率
