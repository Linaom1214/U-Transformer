import os
import shutil
from PIL import Image
from PIL import ImageFilter
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from utils.utils import decode_bbox
from utils.utils import findnearest
from nets.centernet import CenterNet_Swin, CenterNet_Resnet50

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


class EvaluteDatasets(Dataset):
    def __init__(self, test_path):
        super(EvaluteDatasets, self).__init__()
        self.test_path = test_path
        self.img_path = open(self.test_path).readlines()

    def __getitem__(self, index):
        data = self.img_path[index].split()
        file_name = data[0]
        image = Image.open(file_name)
        image = image.convert('RGB')
        # image = image.filter(ImageFilter.SMOOTH)
        image = np.array(image) / 256
        image = np.transpose(image, (2, 0, 1))
        _, file_name = os.path.split(file_name)
        image_id = file_name.split('.')[0]
        return image_id, torch.from_numpy(image)

    def __len__(self):
        return len(self.img_path)


class CenterNet(object):
    def __init__(self, model, **kwargs):
        self.centernet = model
        self.class_names = self._get_class()
        self.confidence = 0.5
        self.nms_threhold = 0.5
        self.image_size = (256, 256)

    @classmethod
    def _get_class(self):
        classes_path = os.path.expanduser('data/classes.txt')
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


class mAP_CenterNet(CenterNet):
    def detect_batch_image(self, dl):
        self.centernet = self.centernet.eval().cuda()
        f = open("./input/detection-results/detection-results.txt", "w")
        f.write("Frame %s" % str(len(dl.dataset)))
        f.write("\n")
        with torch.no_grad():
            for img_ids, image in tqdm(dl):
                image = image.type(torch.FloatTensor).cuda()
                hms, offsets = self.centernet(image)
                for i in range(hms.size(0)):
                    outputs = decode_bbox(hms[i].unsqueeze(0), offsets[i].unsqueeze(0), self.image_size,
                                          self.confidence)
                    output = outputs[0]
                    if len(output) <= 0:
                        f.write("\n")
                    else:
                        batch_boxes, det_conf, det_label = output[:, :2], output[:, 2], output[:, 3]
                        det_xmin, det_ymin, = batch_boxes[:, 0], batch_boxes[:, 1]  # 中心点
                        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
                        top_label_indices = det_label[top_indices].tolist()
                        top_xmin, top_ymin = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
                            det_ymin[top_indices], -1)

                        boxes = np.concatenate([
                            top_xmin[:, 0:1],
                            top_ymin[:, 0:1],
                        ], axis=-1)
                        boxes *= 256
                        for i, c in enumerate(top_label_indices):
                            xc, yc = boxes[i]
                            f.write("%s %s " % (str(int(xc)), str(int(yc))))
                        f.write("\n")
            f.close()


class VocEvaluate(object):
    def __init__(self, test_annos='data/test2022.txt'):
        self.test_annos = test_annos
        self.__check_input()
        ds = EvaluteDatasets(test_annos)
        self.dataloader = DataLoader(ds, batch_size=32, shuffle=False)

    def pred(self, model):
        if not os.path.exists("./input/detection-results"):
            os.makedirs("./input/detection-results")
        else:
            shutil.rmtree("./input/detection-results")
            os.makedirs("./input/detection-results")
        centernet = mAP_CenterNet(model)
        centernet.detect_batch_image(self.dataloader)  # dataloader 

    def get_map(self):
        txt_gt_path = "./input/ground-truth/ground-truth.txt"
        txt_generate_path = "./input/detection-results/detection-results.txt"

        fid1 = open(txt_gt_path, 'r')  # get ground truth txt
        txt_gt = fid1.readlines()
        fid1.close()

        fid2 = open(txt_generate_path, 'r')  # get ground truth txt
        txt_generate = fid2.readlines()
        fid2.close()

        frame_num = int(txt_gt[0].split()[1])

        right_det_sum = 0
        right_nodet_sum = 0
        miss_sum = 0
        false_sum = 0

        for i in range(frame_num):
            right_det = 0
            right_nodet = 0
            miss = 0
            false = 0

            point_gt_num = len(txt_gt[i + 1].split()) // 2
            point_generate_num = len(txt_generate[i + 1].split()) // 2
            if (point_gt_num == 0 or point_generate_num == 0):
                false = false + max(0, point_generate_num - point_gt_num)
                miss = miss + max(0, point_gt_num - point_generate_num)
            else:
                point_gt_loc = np.zeros(shape=(point_gt_num, 2))
                point_generate_loc = np.zeros(shape=(point_generate_num, 2))

                for k1 in range(point_gt_num):
                    point_gt_loc[k1, 0] = int(txt_gt[i + 1].split()[0 + k1 * 2])
                    point_gt_loc[k1, 1] = int(txt_gt[i + 1].split()[1 + k1 * 2])
                for k2 in range(point_generate_num):
                    point_generate_loc[k2, 0] = int(txt_generate[i + 1].split()[0 + k2 * 2])
                    point_generate_loc[k2, 1] = int(txt_generate[i + 1].split()[1 + k2 * 2])

                for k3 in range(point_gt_num):
                    if len(point_generate_loc) > 0:
                        for k4 in range(point_generate_num):
                            eraseID = -1
                            Id1, point1 = findnearest(point_gt_loc[k3], point_generate_loc)
                            Id2, point2 = findnearest(point1, point_gt_loc)
                            if (Id2 == k3):
                                eraseID = Id1
                                deltax = abs(point_gt_loc[k3, 0] - point1[0])
                                deltay = abs(point_gt_loc[k3, 1] - point1[1])
                                if ((deltax <= 1.5) and (deltay <= 1.5)):
                                    right_det = right_det + 1
                                elif ((deltax <= 4.5) and (deltay <= 4.5)):
                                    right_nodet = right_nodet + 1
                                else:
                                    miss = miss + 1
                                    false = false + 1
                            else:
                                miss = miss + 1
                            if (eraseID != -1):
                                point_generate_loc = np.delete(point_generate_loc, eraseID, axis=0)
                                break
                false = false + len(point_generate_loc)

            right_det_sum = right_det_sum + right_det
            right_nodet_sum = right_nodet_sum + right_nodet
            miss_sum = miss_sum + miss
            false_sum = false_sum + false

        Recall = right_det_sum / (miss_sum + right_det_sum)
        Precision = right_det_sum / (right_det_sum + right_nodet_sum + 1e-5)
        F1 = 2 * (Precision * Recall) / (Precision + Recall + 1e-5)
        score_sum = right_det_sum * 1 - miss_sum * 1 - false_sum * 2

        return F1, Recall, Precision

    def __check_input(self):
        image_ids = open(self.test_annos).readlines()
        if not os.path.exists("./input"):
            os.makedirs("./input")
        else:
            shutil.rmtree("./input")
            os.makedirs("./input")
        if not os.path.exists("./input/ground-truth"):
            os.makedirs("./input/ground-truth")
        else:
            shutil.rmtree("./input/ground-truth")
            os.makedirs("./input/ground-truth")

        f = open("./input/ground-truth/ground-truth.txt", "w")

        f.write("Frame %d" % len(image_ids))
        f.write("\n")

        for path in image_ids:
            data = path.split()
            info_data = data[1:]
            if len(info_data) != 0:
                for info in info_data:
                    xc, yc, obj_name = info.split(',')
                    f.write("%s %s " % (xc, yc))
            else:
                f.write(" ")
            f.write("\n")
        f.close()

if __name__ == '__main__':
    input_shape = (256, 256, 3)
    classes_path = 'data/classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    backbone = "swim"
    Cuda = True
    test_annos = "./data/test2022.txt"

    assert backbone in ['resnet50', "swim"]
    if backbone == "resnet50":
        model = CenterNet_Resnet50(num_classes)
    else:
        model = CenterNet_Swin(num_classes)

    model_path = 'logs/best.pt'
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict)
    print('Finished!')
    net = model.eval()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    evalor = VocEvaluate(test_annos=test_annos)
    evalor.pred(net)
    F1, Recall, Precision = evalor.get_map()

    print('ReCall  %.3f' % Recall)
    print('Precision  %.3f' % Precision)
    print('F1 Score   %.3f' % F1)
