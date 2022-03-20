import numpy as np
from argparse import ArgumentParser

def findnearest(refdata, detdata):
    dis = np.zeros((len(detdata), 1))
    for i in range(len(detdata)):
        dis[i] = abs(refdata[0] - detdata[i][0]) + abs(refdata[1] - detdata[i][1])
    Id = np.argmin(dis)
    point = detdata[Id]
    return Id, point

def calculate_metric(txt_gt_path, txt_generate_path):
    fid1 = open(txt_gt_path, 'r') 
    txt_gt = fid1.readlines()
    fid1.close()

    fid2 = open(txt_generate_path, 'r') 
    txt_generate = fid2.readlines()
    fid2.close()

    frame_num = int(txt_gt[0].split()[1])

    right_det_sum = 0
    right_nodet_sum = 0
    miss_sum = 0
    false_sum = 0
    fp_sum = 0

    for i in range(frame_num):
        right_det = 0
        right_nodet = 0
        miss = 0
        false = 0

        point_gt_num = len(txt_gt[i + 1].split()) // 2
        point_generate_num = len(txt_generate[i + 1].split()) // 2

        if point_gt_num == 0 or point_generate_num == 0:
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
                        if Id2 == k3:
                            eraseID = Id1
                            deltax = abs(point_gt_loc[k3, 0] - point1[0])
                            deltay = abs(point_gt_loc[k3, 1] - point1[1])
                            if (deltax <= 1.5) and (deltay <= 1.5):
                                right_det = right_det + 1
                            elif (deltax <= 4.5) and (deltay <= 4.5):
                                right_nodet = right_nodet + 1
                            else:
                                miss = miss + 1
                                false = false + 1
                        else:
                            miss = miss + 1
                        if eraseID != -1:
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

    print('ReCall %.3f Precision %.3f F1 %.3f Score %d' % (Recall,Precision,F1,score_sum))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--txt-gt-path', type=str, default="./input/ground-truth/ground-truth.txt", help='txt_gt_path')
    parser.add_argument('--txt-pd-path', type=str, default="input/detection-results/detection-results.txt", help='txt_pd_path')
    calculate_metric(parser.parse_args().txt_gt_path, parser.parse_args().txt_pd_path)