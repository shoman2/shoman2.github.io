---
layout: post
title:  "이미지처리 - Faster RCNN"
subtitle:   "Language Modeling"
categories: data
tags: dl
comments: true
---
# 1.이미지 처리 개요

source: https://herbwood.tistory.com/10?category=856250



## 배경

컴퓨터 비전의 주요 과제 3가지는 다음과 같음

1)분류 (Classification)

2)물체 인식 (Object Detection)

3)이미지 영역분할 (Image Segmentation)

상기 3가지로 기술한 컴퓨터 비전 과제 중 Faster R-CNN 까지는 Object Dtection영역이라면 본 제안내용의 주가될 Mask R-CNN은 Image Segmentation 영역으로 구분



## 이미지 분류 및 물체 인식 알고리즘의 역사

### *1)R-CNN*



### *2)Fast R-CNN*



### *3)Faster R-CNN*

Fast R-CNN에 RPN(region proposal network)를 추가한 구조



### *4)Mask R-CNN*

- Fast R-CNN의 classification, localization(bounding box regression) branch에 새롭게 mask branch가 추가
- RPN 전에 FPN(feature pyramid network)가 추가
- Image segmentation의 masking을 위해 RoI align이 RoI pooling을 대신



N x N 사이즈의 인풋 이미지가 주어졌을때 Mask R-CNN의 process는

**1. 800~1024 사이즈로 이미지를 resize해준다. (using bilinear interpolation)**

**2. Backbone network의 인풋으로 들어가기 위해 1024 x 1024의 인풋사이즈로 맞춰준다. (using padding)**

**3. ResNet-101을 통해 각 layer(stage)에서 feature map (C1, C2, C3, C4, C5)를 생성한다.**

**4. FPN을 통해 이전에 생성된 feature map에서 P2, P3, P4, P5, P6 feature map을 생성한다.**

**5. 최종 생성된 feature map에 각각 RPN을 적용하여 classification, bbox regression output값을 도출한다.**

**6. output으로 얻은 bbox regression값을 원래 이미지로 projection시켜서 anchor box를 생성한다.**

**7. Non-max-suppression을 통해 생성된 anchor box 중 score가 가장 높은 anchor box를 제외하고 모두 삭제한다.**

**8. 각각 크기가 서로다른 anchor box들을 RoI align을 통해 size를 맞춰준다.**

**9. Fast R-CNN에서의 classification, bbox regression branch와 더불어 mask branch에 anchor box값을 통과시킨다.**



Mask R-CNN에서는 backbone으로 ResNet-101을 사용하는데 ResNet 네트워크에서는 이미지 input size가

800~1024일때 성능이 좋다고 알려져있다. (VGG는 224 x 224)

따라서 이미지를 위 size로 맞춰주는데 이때 **bilinear interpolation**을 사용하여 resize해준다.



bilinear interpolation은 여러 interpolation기법 중 하나로 동작과정은 다음과 같다.

2 x 2의 이미지를 위 그림과 같이 4 x 4로 Upsampling을 한다면 2 x 2에 있던 pixel value가 각각

P1, P2, P3, P4로 대응된다. 이때 총 16개 중 4개의 pixel만 값이 대응되고 나머지 12개는 값이 아직 채워지지 않았는데 이를 bilinear interpolation으로 값을 채워주는 것이다. 계산하는 방법은 아래와 같다.



이렇게 기존의 image를 800~1024사이로 resize해준 후 네트워크의 input size인 1024 x 1024로 맞춰주기 위해

나머지 값들은 **zero padding**으로 값을 채워준다.



