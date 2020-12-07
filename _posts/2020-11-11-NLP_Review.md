---
layout: post
title:  "자연어처리 - NLG 관련 기존 내용 리뷰"
subtitle:   "Review: Statistical & Geometric Perspective for Deep Learning"
categories: data
tags: dl
comments: true
---

# Review: Statistical & Geometric Perspective for Deep Learning
- 기존 Our objective is:

  - 세상에 존재하는 어떠한 미지의 함수를 찾자 (Approximation)

- 주어진 입력값 x에 대해서 원하는 출력 y를 반환하도록, 손실함수를 최소화하는 파라메터(theta)를 찾자

  - Loss Function/Loss 를 Minize하기위해 ->  Gradient descent를 수행하기위해 back-propagation을 수행하자

  ---

  

- 추 후에는 Our objective becomes...

  - 세상에 존재하는 어떤 미지의 확률 분포 함수를 모사하자.(Approximation)

- 플러스, **Probablistic** Perspective

  - 확률 분포 P(x)와 P(y|x)로부터 데이터를 수집하여,
  - 해당 데이터를 가장 잘 설명하는 확률 분포 함수의 파라미터(Theta)를 찾자: logP(y|x;theta)
    - MLE(확률분포함수를 근사하고 싶기때매..)
    - 쎄타를 바꾸어가면서.. Gradient Desent using Back-propagation
  - 또는 두 확률 분포를 비슷하게 만들자
    - Minimize Cross Entropy (or KL-Divergence)

- Neural Net도 확률분포 함수다! 그리고 NN의 파라메터는 해당 확률분포의 파라메터다.

- 플러스, **Geometric** Perspective(기하학적 관점)

  - 데이터란 저차원의 manifold에 분포하고 있으며, 여기에 약간의 노이즈 e(ta)가 추가되어 있는 것
    - 노이즈란 task(x->y)에 따라서 다양하게 해석 가능 할 것
  - 따라서, 해당 manifold를 배울 수 있다면, 더 낮은 차원으로 효율적인 맵핑이 가능
    - non-linear dimension reduction (ex. Autoencoder)

- **Reprentation Learning**, Again
  
  - 낮은 차원으로의 표현을 통해, 차원의 저주를 벗어나 효과적인 학습이 가능

# Review: Auto Encoders

- Encoder와 Decoder를 통해 압축과 해제를 실행하는 것이 오토 인코더
  - 인코더는 입력(x)의 정보를 최대한 보존하도록 손실 압축을 수행
  - 디코더는 중간 결과물(z)의 정보를 입력(x)과 같아지도록 압축해제(복원)를 수행
- 복원을 성공적으로 하기 위해서, 오토인코더는 Feature를 추출하는 방법을 자동으로 학습
  - 이게 왜 중요한가 ? 워드 임베딩 / Text Classification 이 이와 일맥상통한다
  - In Word2Vec
    - Objective: 주어진 단어로 주변 단어를 예측하자
    - y를 예측하기 위해 필요한 정보가 z에 있어야 한다.
      - 주변 단어를 잘 예측하기 위해 x를 잘 압축하자.
  - In Text Classification
    - 1)Using RNN
      - Word의 시퀀스로부터 정보를 잘 찾아내야함. 
      - 즉 RNN은 하나의 벡터로 센텐스나 컨텍스트를 Embedding한다.
    - 2)Using CNN
      - 문장 임베딩 벡터를 뽑아낸다..
      - Latent Space에서 Decision Boundary를 찾는점은 다 동일..
- 정리
  - 신경망은 x와 y사이의 관계를 학습하는 과정에서 feature를 자연스럽게 학습
    - 특히 저차원으로 축소(압축)되는 과정에서 정보의 취사/선택이 이루어짐
  - Word Embedding(Skip-gram):
    - 주변 단어( y)를 예측하기 위해 필요한 정보를 현재 단어(x)에서 추출하여 압축
  - Sentence Embedding(text classification):
    - Label(y)을 예측하기 위해 필요한 정보를 단어들의 시퀀스( x)로부터 추출하여 압축

# Intro to NLG

- 컴퓨터가 인간이 만들어놓은 대량의 문서를 통해 정보를 얻고(NL Understanding)
- **얻어낸 정보를 사람이 이해할 수 있게 사람의 언어로 표현하는 것(NL Generation)**
- Seq-to-Seq 이전에..
  - 1)Word Embedding[Mikolov et al., 2013]
  - Text Classification[Kim, 2014]
    - --> 결국 한다고 하는 것은 Text to Numeric Values였다..
- Seq-to-Seq with Attention의 전파 ! 이후에는?
  - Beyond "text to numeric"
  - Numeric벨류를 Text로 다시 바꾸어 낼 수 있는지에 대한 공부를 해보자..
-  Era of Attention
  - Transformer의 등장으로 인해 연구는 더더욱 빨라짐 ()
    - PLM(Pretrained Language Model)의 유행으로 인해 NLG뿐만 아니라 NLP의 다른 영역에도 큰 영향을 줌
    - 거스를 수 없는 대세다! PLM기억해야한다.
      - 흐름 : **Language Model->Seq to Seq with Attention -> Transformer -> PLM -> Advanced NLP**
    - Neural Machine Translation을 통해 자연어 생성의 근본부터 단단히 다질 수 있도록 배워나가자
      - 반복된 학습을 통해 Auto-regressive 특성을 몸으로 익히고,
      - 이를 해결하기 위해 여러가지 (Empirical + Mathmatical)방법들을 다룬다
    - Seq-to-Seq w/ Attention 뿐만 아니라,  Transformer도 nano 단위로 detail하게 분해하여 이해/구현할 수 있도록 공부하자
    - 이정도 공부하면 추후 PLM을 활용한 NLP 심화 과정을 어려움 없이 스터디할 수 있도록!! 밑바탕을 만들자.
