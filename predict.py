
from PIL import Image
import glob
import cv2
import numpy as np

from centernet import CenterNet

imgs = glob.glob('/home/mat/Desktop/dataset/test/test/*.bmp')


def fun(path):
    return int(path.split('\\')[-1].split('.bmp')[0].split('data')[-1])

imgs.sort(key=fun)
centernet = CenterNet()

VideoOut = False # out put video
PipelineFilter = False # use filter pipeline 

if VideoOut:
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter('output.avi',fourcc, 60.0, (256,256))

for i, img in enumerate(imgs):
    image = Image.open(img)
    image = image.convert('RGB')
    if PipelineFilter:
        r_image = centernet.detect_piple(image)
    else:
        r_image = centernet.detect_image(image)
    image_name = img.split('/')[-1].split('.')[0]
    r_image = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', r_image)
    if VideoOut:
        out.write(r_image)
    cv2.waitKey(5)

if VideoOut:
    out.release()
cv2.destroyAllWindows()
