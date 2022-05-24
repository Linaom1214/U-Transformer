import colorsys
import os
import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from torch import nn
from torch.autograd import Variable
from math import sqrt

from nets.centernet import CenterNet_Resnet50, CenterNet_Swin
from utils.utils import  decode_bbox


# create piple
class Piple(object):
    def __init__(self):
        super().__init__()
        self.length = 0
        self.data = []
        self.state = False

    def std(self):
        return np.std(self.data, axis=0)

    def __call__(self):
        if len(self.data) > 3:
            self.state = True

# mangae piple
class AdaptPiple(object):
    def __init__(self):
        super().__init__()
        self.piple = []

    def addpiple(self, Piple):  # add piple
        self.piple.append(Piple)

    def subpiple(self, i):  # del piple
        del self.piple[i]

    def numpiple(self):  # piple length
        return len(self.piple)



def preprocess_image(image):
    return np.float32(image) / 255.


class CenterNet(object):
    _defaults = {
        "model_path": 'logs/best.pt',
        "classes_path": 'data/classes.txt',
        "backbone": "swin",
        "image_size": [256, 256, 3],
        "confidence": 0.5,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()
        self.R0 = 5
        self.data = []
        self.adapt = AdaptPiple()
        self.frame = 0

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def generate(self):
        self.num_classes = len(self.class_names)


        assert self.backbone in ['resnet50', 'swin']
        if self.backbone == "resnet50":
            self.centernet = CenterNet_Resnet50(num_classes=self.num_classes)
        else:
            self.centernet = CenterNet_Swin(num_classes=self.num_classes)
        print('Now Using %s'%self.backbone)


        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = self.centernet.state_dict()
        pretrained_dict = torch.load(self.model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        self.centernet.load_state_dict(model_dict)
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.centernet = nn.DataParallel(self.centernet)
            self.centernet.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = image.resize((self.image_size[0], self.image_size[1]))

        photo = np.array(crop_img, dtype=np.float32)[:, :, ::-1]

        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)),
                           [1, self.image_size[2], self.image_size[0], self.image_size[1]])

        with torch.no_grad():
            images = Variable(torch.from_numpy(np.asarray(photo)).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            outputs = self.centernet(images)

            outputs = decode_bbox(outputs[0], outputs[1], self.image_size, self.confidence)

            output = outputs[0]
            if len(output) <= 0:
                return image
            batch_boxes, det_conf, det_label = output[:, :2], output[:, 2], output[:, 3]
            det_xmin, det_ymin, = batch_boxes[:, 0], batch_boxes[:, 1]  

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(det_ymin[top_indices], -1)

            boxes = np.concatenate([
                top_xmin[:, 0:1],
                top_ymin[:, 0:1],
            ], axis=-1)
            boxes *= image_shape

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            top, left = boxes[i]

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label = label.encode('utf-8')
            print(label, top, left)

            draw.ellipse([int(top - 2), int(left - 2), int(top + 2), int(left + 2)], outline='red')

            del draw
        return image


    def detect_piple(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        crop_img = image.resize((self.image_size[0], self.image_size[1]))
        photo = np.array(crop_img, dtype=np.float32)[:, :, ::-1]
        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)),
                           [1, self.image_size[2], self.image_size[0], self.image_size[1]])

        with torch.no_grad():
            images = Variable(torch.from_numpy(np.asarray(photo)).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            outputs = self.centernet(images)


            outputs = decode_bbox(outputs[0], outputs[1], self.image_size, self.confidence)
            output = outputs[0]
            if len(output) <= 0:
                return image
            batch_boxes, det_conf, det_label = output[:, :2], output[:, 2], output[:, 3]
            det_xmin, det_ymin, = batch_boxes[:, 0], batch_boxes[:, 1]  

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(det_ymin[top_indices], -1)

            boxes = np.concatenate([
                top_xmin[:, 0:1],
                top_ymin[:, 0:1],
            ], axis=-1)
            boxes *= image_shape
        if self.frame < 5:
            self.data.append(boxes)
        elif self.frame == 5:
            self.init_piple(self.data, self.adapt)
        else:
            # 第一遍遍历 将当前坐标信息填入管道
            num_piple = self.adapt.numpiple()
            for i in range(num_piple):
                stdx, stdy = self.adapt.piple[i].std()
                av = np.mean([stdx, stdy])
                for j, (x, y) in enumerate(boxes):
                    xc, yc = np.mean(self.adapt.piple[i].data, axis=0)  # 当前管道内的平均坐标值
                    if (sqrt((xc - x) ** 2) < self.R0 + 3 * av) and (abs(yc - y) < self.R0 + 3 * av):  # 判断目标是否在管道内 而且要求目标是运动的
                        self.adapt.piple[i].length += 1
                        del self.adapt.piple[i].data[0]  # 删除第一个存入的数据信息
                        self.adapt.piple[i].data.append(np.array([x, y]))  # 当前管道数据更新
                    else:
                        self.adapt.piple[i].length -= 1
                        self.adapt.addpiple(Piple())  # 目标不在当前管道中 当前管道长度减一 并由此创建新的管道
                        self.adapt.piple[-1].data.append(np.array([x, y]))
                        self.adapt.piple[-1].length += 1
                if self.adapt.piple[i].length >= 5:
                    self.adapt.piple[i].length = 5
                if self.adapt.piple[i].length < 3:
                    self.adapt.subpiple(i)
                    break
        self.frame += 1

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.image_size[0], 1)

        if self.frame <= 5:
            for i, c in enumerate(top_label_indices):
                predicted_class = self.class_names[int(c)]
                score = top_conf[i]
                top, left = boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))

                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                # print(label, top, left)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                draw.ellipse([int(top - 2), int(left - 2), int(top + 2), int(left + 2)], outline='red')
                del draw
        else:
            num_piple = self.adapt.numpiple()
            for i, c in enumerate(range(num_piple)):
                if len(self.adapt.piple[i].data) >= 2:
                    top, left = self.adapt.piple[i].data[-1]
                    top = max(0, np.floor(top).astype('int32'))
                    left = max(0, np.floor(left).astype('int32'))
                    draw = ImageDraw.Draw(image)
                    draw.ellipse([int(top - 2), int(left - 2), int(top + 2), int(left + 2)], outline='red')
                    # del draw
                else:
                    continue
            for i, c in enumerate(top_label_indices):
                predicted_class = self.class_names[int(c)]
                score = top_conf[i]
                top, left = boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))

                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                # print(label, top, left)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                draw.ellipse([int(top - 4), int(left - 4), int(top + 4), int(left + 4)], outline='blue')
                del draw

        return image

    def init_piple(self,data, adapt):
        data = np.array(data)
        num_target = len(data[0])
        for _ in range(num_target):
            adapt.addpiple(Piple())  # 根据目标数实例对象
        for i in range(num_target):
            adapt.piple[i].data.append(data[0][i])  # 数据暂存到管道中
            xc, yc = data[0][i][0], data[0][i][1]  # 第一帧的目标位置
            for j in range(1, 5):
                data_ = data[j]
                for box in data_:
                    x, y = box
                    if (abs(xc - x) < self.R0) and (abs(yc - y) < self.R0):  # 判断目标是否在管道内
                        adapt.piple[i].length += 1
                        xc, yc = x, y
                        adapt.piple[i].data.append(box)
            if adapt.piple[i].length < 3:
                adapt.subpiple(i)
