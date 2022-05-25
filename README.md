# Infrared Dim Small Target Detection Based On U-Transformer

![](https://img.shields.io/badge/Python-3.8%2B-red)
![](https://img.shields.io/badge/Pytorch-1.10%2B-brightgreen)
![](https://img.shields.io/badge/Vision_Transformer-brightgreen)
![](https://img.shields.io/badge/Infrared_Small_Dim_Target_Detection-yellow)

<img src="./src/fig2.png" width = "8000" height = "300"/>
<img src="./src/fig3.png" width = "800" height = "500"  align=center/>

## Update 2022.5.24 Fix training result not same with paper 

## Notice !

When using Nvidia P100 select pytorch version <1.8> !


## Colab Examples

<a href="https://colab.research.google.com/drive/12ZQ8l3WUMVgA4Qfa6tTTUdobnyhRftcG?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>



## ABSTRACT

Infrared dim and small target detection is a key technology for space-based infrared search and tracking systems. The traditional detection algorithms have a high false alarm rate and fail to handle complex background and high-noise scenarios. Also, the algorithms cannot effectively detect targets on a small scale. In this paper, a U-Transformer based on Center Net is proposed, and the Swin Transformer is introduced into the infrared dim small target detection algorithm for the first time. First, a U-shaped network is constructed. In the encoder part, the self-attention mechanism of the Swin transformer is used for infrared dim and small target feature extraction, which helps to solve the problems of losing dim and small target features in deep networks. Meanwhile, by using the encoding and decoding structure, infrared dim and small target features are filtered from the complex background while the shallow features and semantic information of the target are retained. Experiments show that the anchor free and the Swin transformer have great potential for infrared dim small target detection. On the datasets with a complex background, our algorithm outperforms the state-of-the-art algorithms and meets the real-time requirement.

## [Datasets](#Infrared-Dim-Small-Target-Detection-Based-On-U-Transformer)

[A dataset for infrared detection and tracking of dim-small aircraft targets under ground / air background](http://www.csdata.org/p/387/)

### DATA Prepare

```text
train.txt (store train data)
test.txt (store test data)
```


## [Usage](#Infrared-Dim-Small-Target-Detection-Based-On-U-Transformer)

### Train

Define Configuration in train.py
```python 
    # ----------------------------------------#
    # Configuration
    input_shape = (256, 256, 3)
    classes_path = 'data/classes.txt'
    annotation_path = 'data/train2022.txt'
    lr = 1e-3
    Batch_size = 16
    Init_Epoch = 0
    Total_Epoch = 50
    backbone = "swin"
    Cuda = True
```
training
```Python
python train.py
```
### Inference

Define Model information in centernet.py
```python
    _defaults = {
        "model_path": 'logs/best.pt',
        "classes_path": 'data/classes.txt',
        "backbone": "swin",
        "image_size": [256, 256, 3],
        "confidence": 0.5,
        "cuda": True
    }
```

```python
python predict.py
```
### Evaluation

Define Model information in centernet.py
```python
    _defaults = {
        "model_path": 'logs/best.pt',
        "classes_path": 'data/classes.txt',
        "backbone": "swin",
        "image_size": [256, 256, 3],
        "confidence": 0.5,
        "cuda": True
    }
```

```
python eval.py
```
## [Results](#Infrared-Dim-Small-Target-Detection-Based-On-U-Transformer)

| Method               | Recall | Precision | F1    | Score    | FPS | Platform      | Language      |
|----------------------|--------|-----------|-------|----------|-----|---------------|---------------|
| **U-Transformer** | 0.778  | 0.995     | 0.873 | 2774     | 65  | GTX 1080TI    | Python        |
| YOLOX (2021)         | 0.79   | 0.97      | 0.871 | 1768     | 24  | GTX 1080TI    | Python        |
| DANet (2021)         | 0.815  | 0.912     | 0.86  | -34031   | 32  | GTX 1080TI    | Python        |
| CenterNet (2019)     | 0.728  | 0.876     | 0.795 | 363      | 147 | GTX 1080TI    | Python        |
| RLCM (2018)          | 0.444  | 0.949     | 0.605 | -813938  | 32  | Core i7-7700K | Python+Matlab |
| TopHat (-)           | 0.554  | 0.603     | 0.577 | -161645  | 107 | Core i7-7700K | Python+Matlab |
| ADMD (2020)          | 0.455  | 0.754     | 0.567 | -1082416 | 38  | Core i7-7700K | Python+Matlab |
| TLLICM (2019)        | 0.347  | 0.931     | 0.506 | -144027  | 30  | Core i7-7700K | Python+Matlab |
| YOLOV5 (2020)        | 0.291  | 0.978     | 0.449 | -4070    | 100 | GTX 1080TI    | Python        |
| AGPCNET              | 0.063  | 0.159     | 0.090 | -29455   |  2  | Tesla T4      | Python        |
## [TODO](#Infrared-Dim-Small-Target-Detection-Based-On-U-Transformer)

![](https://img.shields.io/badge/TensorRT_Deploy-blue)

## Bugs 
### nvidia P100 traning very slow



