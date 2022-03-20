from random import shuffle
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import draw_gaussian


def preprocess_image(image):
    return np.float32(image) / 255.


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class CenternetDataset(Dataset):
    def __init__(self, train_lines, input_size, num_classes, is_train):
        super(CenternetDataset, self).__init__()

        self.train_lines = train_lines
        self.input_size = input_size
        self.output_size = (int(input_size[0] / 1), int(input_size[1] / 1))  # 2倍下采样
        self.num_classes = num_classes
        self.is_train = is_train

    def __len__(self):
        return len(self.train_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])  # 仅保存 中心点  x , y 信息
        if not random:
            # resize image
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box), 3))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0]] = box[:, [0]] * nw / iw + dx
                box[:, [1]] = box[:, [1]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box_data = np.zeros((len(box), 3))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0]] = box[:, [0]] * nw / iw + dx
            box[:, [1]] = box[:, [1]] * nh / ih + dy
            if flip: box[:, [0]] = w - box[:, [0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box_data = np.zeros((len(box), 3))
            box_data[:len(box)] = box

        return image_data, box_data

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        img, y = self.get_random_data(lines[index], [self.input_size[0], self.input_size[1]], random=True)
        batch_hm = np.zeros((self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
        batch_reg = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
        if len(y) != 0:
            boxes = np.array(y[:, :2], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.input_size[1] * self.output_size[1]
            boxes[:, 1] = boxes[:, 1] / self.input_size[0] * self.output_size[0]
        for i in range(len(y)):
            bbox = boxes[i].copy()
            bbox = np.array(bbox)
            bbox[0] = np.clip(bbox[0], 0, self.output_size[1] - 1)
            bbox[1] = np.clip(bbox[1], 0, self.output_size[0] - 1)
            cls_id = int(y[i, -1])
            radius = 3 
            ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
            batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
            batch_reg_mask[ct_int[1], ct_int[0]] = 1
        img = np.array(img, dtype=np.float32)[:, :, ::-1]
        img = np.transpose(preprocess_image(img), (2, 0, 1))
        return img, batch_hm, batch_reg, batch_reg_mask


# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_regs, batch_reg_masks = [], [], [], []

    for img, batch_hm, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs = np.array(imgs)
    batch_hms = np.array(batch_hms)
    batch_regs = np.array(batch_regs)
    batch_reg_masks = np.array(batch_reg_masks)
    return imgs, batch_hms, batch_regs, batch_reg_masks
