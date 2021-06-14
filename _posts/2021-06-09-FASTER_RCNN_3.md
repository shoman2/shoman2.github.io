---
layout: post
title:  "이미지처리 - Faster RCNN 3"
subtitle:   "Language Modeling"
categories: data
tags: dl
comments: true
---
# RPN(Region Proposal network)



RPN은 원본 이미지에서 region proposals를 추출하는 네트워크입니다. 원본 이미지에서 anchor box를 생성하면 수많은 region proposals가 만들어집니다.

RPN은 region proposals에 대하여 **class score**를 매기고, bounding **box coefficient를 출력**하는 기능을 합니다.

1. 원본 이미지를 pre-trained된 VGG 모델에 입력하여 feature map을 얻습니다.

원본 이미지의 크기가 800x800이며, sub-sampling ratio가 1/100이라고 했을 때 8x8 크기의 feature map이 생성됩니다(channel 수는 512개 입니다).

1. 위에서 얻은 feature map에 대하여 3x3 conv 연산을 적용합니다. 이때 feature map의 크기가 유지될 수 있도록 padding을 추가합니다.

(H + 2P -FH)/S   + 1  = output size

(W+2P-FW)/S   + 1  = output size

8x8x512 feature map에 대하여 3x3 연산을 적용하여 8x8x512개의 feature map이 출력됩니다., padding = 1

stride = 1

(8+2-3)/1 +1 = 8

1. class score를 매기기 위해서 feature map에 대하여 1x1 conv 연산을 적용합니다. 이 때 출력하는 feature map의 channel 수가 2x9가 되도록 설정합니다. RPN에서는 후보 영역이 어떤 class에 해당하는지까지 구체적인 분류를 하지 않고 객체가 포함되어 있는지 여부만을 분류합니다. 또한 anchor box를 각 grid cell마다 9개가 되도록 설정했습니다. 따라서 channel 수는 2(object 여부) x 9(anchor box 9개)가 됩니다. 8x8x512 크기의 feature map을 입력받아 8x8x2x9크기의 feature map을 출력합니다.
2. bounding box regressor를 얻기 위해 feature map에 대하여 1x1 conv 연산을 적용합니다. 이 때 출력하는 feature map의 channel 수가 **4(bounding box regressor)x9(anchor box 9개)**가 되도록 설정합니다.

8x8x512 크기의 feature map을 입력받아 8x8x4x9크기의 feature map을 출력합니다.\



# 3. Multi-task loss

- i : mini-batch 내의 anchor의 index

- pi : anchor i에 객체가 포함되어 있을 예측 확률

- p∗i : anchor가 양성일 경우 1, 음성일 경우 0을 나타내는 index parameter

- ti : 예측 bounding box의 파라미터화된 좌표(coefficient)

- t∗i : ground truth box의 파라미터화된 좌표

- Lcls : Loss loss

- Lreg : Smooth L1 loss

- Ncls : mini-batch의 크기(논문에서는 256으로 지정)

- Nreg : anchor 위치의 수

- λ : balancing parameter(default=10)

  RPN과 Fast R-CNN을 학습시키기 위해 Multi-task loss를 사용합니다.

  하지만 **RPN에서는 객체의 존재 여부만을 분류**하는 반면, **Fast R-CNN에서는 배경을 포함한 class를 분류**한다는 점에서 차이가 있습니다.



# 4.Training Faster R-CNN

### 1.feature extraction by pre-trained VGG16

**pre-trained된 VGG16 모델에 800x800x3** 크기의 원본 이미지를 입력하여 **50x50x512** 크기의 feature map을 얻습니다. 여기서 sub-sampling ratio는 1/16입니다.

- **Input** : 800x800x3 sized image
- **Process** : feature extraction by pre-trained VGG16
- **Output** : 50x50x512 sized feature map

### 2.Generate Anchors by Anchor generation layer

region proposals를 추출하기에 앞서 원본 이미지에 대하여 anchor box를 생성하는 과정이 필요합니다. **원본 이미지의 크기에 sub-sampling ratio를 곱한만큼의 grid cell이 생성** 되며, 이를 기준으로 각 grid cell마다 9개의 anchor box를 생성합니다.

즉, 원본 이미지에 50x50(=800x1/16 x 800x1/16)개의 grid cell이 생성되고, 각 grid cell마다 9개의 anchor box를 생성하므로 **총 22500(=50x50x9)개의 anchor box**가 생성됩니다.

- **Input** : 800x800x3 sized image
- **Process** : generate anchors
- **Output** : 22500(=50x50x9) anchor boxes

### 3.Class scores and Bounding box regressor by RPN

VGG16으로부터 feature map을 입력받아 anchor에 대한 class score, bounding box regressor를 반환.

- **Input** : 50x50x512 sized feature map
- **Process** : Region proposal by RPN
- **Output** : class scores(50x50x2x9 sized feature map) and bounding box regressors(50x50x4x9 sized feature map)

### 4.Region proposal by Proposal layer

Proposal layer에서는 2)번 과정에서 생성된 anchor boxes와 RPN에서 반환한 class scores와 bounding box regressor를 사용하여 region proposals를 추출하는 작업을 수행

Non maximum suppression을 적용하여 부적절한 객체를 제거한 후, class score 상위 N개의 anchor box를 추출 이후 regression coefficients를 anchor box에 적용하여 anchor box가 객체의 위치를 더 잘 detect하도록 조정.

- Input
  - 22500(=50x50x9) anchor boxes
  - class scores(50x50x2x9 sized feature map) and bounding box regressors(50x50x4x9 sized feature map)
- **Process** : region proposal by proposal layer
- **Output** : top-N ranked region proposals

### 5.Select anchors for training RPN by Anchor target layer

Anchor target layer의 목표는 RPN이 학습하는데 사용할 수 있는 anchor를 선택하는 것2

2번 과정에서 생성한 anchor box 중에서 원본 이미지의 경계를 벗어나지 않는 anchor box를 선택 후 positive/negative 데이터를 sampling.

여기 positive sample은 객체가 존재하는 foreground, negative sample은 객체가 존재하지 않는 background를 의미

전체 anchor box 중에서 1) ground truth box와 가장 큰 IoU 값을 가지는 경우  2) ground truth box와의 IoU 값이 0.7 이상인 경우에 해당하는 box를 positive sample로 선정

반면 ground truth box와의 IoU 값이 0.3 이하인 경우에는 negative sample로 선정

IoU 값이 0.3~0.7인 anchor box는 무시

이러한 과정을 통해 RPN을 학습시키는데 사용할 데이터셋을 구성

- **Input** : anchor boxes, ground truth boxes
- **Process** : select anchors for training RPN
- **Output** : positive/negative samples with target regression coefficients

### 6.Select anchors for training Fast R-CNN by Proposal Target Layer

Proposal target layer의 목표는 proposal layer에서 나온 region proposals 중에서 Fast R-CNN 모델을 학습시키기 위한 유용한 sample을 선택하는 것.

여기서 선택된 region proposals는 1)번 과정을 통해 출력된 feature map에 RoI pooling을 수행

먼저 region proposals와 ground truth box와의 IoU를 계산하여 0.5 이상일 경우 positive, 0.1~0.5 사이일 경우 negative sample로 label

- **Input** : top-N ranked region proposals, ground truth boxes
- **Process** : select region proposals for training Fast R-CNN
- **Output** : positive/negative samples with target regression coefficients

### 7.Max Pooling by RoI pooling

원본 이미지를 VGG-16 모델에 입력하여 얻은 feature map과 6번 과정을 통해 얻은 sample을 사용하여 Region of Interest(RoI) Pooling을 수행

이를 통해 고정된 크기의 feature map이 출력

- Input
  - 50x50x512 sized feature map
  - positive/negative samples with target regression coefficients
- **Process** : RoI pooling
- **Output** : 7x7x512 sized feature map

What is RoI Pooling ?

RoI(Region of Interest) pooling은 feature map에서 region proposals에 해당하는 관심 영역(Region of Interest)을 지정한 크기의 grid로 나눈 후 max pooling을 수행하는 방법입니다. 각 channel별로 독립적으로 수행하며, 이 같은 방법을 통해 고정된 크기의 feature map을 출력하는 것이 가능합니다.

### 8.Train Fast R-CNN by Multi-task loss

나머지 과정은 Fast R-CNN 모델의 동작 순서와 동일.

입력받은 feature map을 fully convolution layer에 입력하여 4096크기의 feature vector를 얻고

이 후 feature vector를 Classifier 및 BBox Regressor에 입력하여 (K+1), (K+1) x 4 크기의 feature vector를 출력

이 출력된 결과를 사용하여 Multi-task loss를 통해 Fast R-CNN 모델을 학습

- **Input** : 7x7x512 sized feature map
- Process
  - feature extraction by fc layer
  - classification by Classifier
  - bounding box regression by Bounding box regressor
  - Train Fast R-CNN by Multi-task loss
- **Output** : loss(Loss loss + Smooth L1 loss)



source : https://herbwood.tistory.com/

