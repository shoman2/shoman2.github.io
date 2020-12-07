---
layout: post
title:  "자연어처리 - N-GRAM 정리 "
subtitle:   "Language Modeling"
categories: data
tags: dl
comments: true
---
# n-gram 정리
n-gram 알고리즘에 대해 다시 정리합니다!

- #### 장점

  - 쉽게 대형 시스템에 적용 가능하다. 즉 Scalable 하다.
  - n-gram 훈련 및 추론 방식이 굉장히 쉽고 간편하다!

- #### **단점**

  - Poor generalization: 등장하지 않은 단어 조합에 대처가 미흡하다.
    - 단어를 discrete symbol로 취급하기 때문
    - 따라서, 비슷한 단어에 대한 확률을 레버리지하지 못함
    - Smoothing과 Back-off방식을 통해서 단점을 보완하려 했으나, 근본적인 해결책이 되지는 못하더라.
  -  Poor with long dependency: 거리가 멀리 떨어져 있는 단어에 대해서 대처가 불가
  - n이 커질수록 연산량 및 용량도 커짐

- #### 실제로 어플리케이션 적용(ASR, SMT)에 있어서 큰 과제다.

- #### 따라서 이런 이슈들을 해결하기위한 Deep Learning의 시대에 돌입하였다..! 뉴럴넷이 갖은 위력들을 알아봅시다.

  # Neural Net LM

- #### Resolve Sparsity

  - Training set
    - [고양이는 좋은 반려동물 입니다.]
  - Test set
    - [강아지는 훌륭한 애완동물 입니다.]  <-- Unseen Word Sequence

- #### n-gram cannot resolve this issue because words are discrete symbols in n-gram model.

- #### Structure

  - ![image-20201113232525636](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20201113232525636.png)
  - Markov Assumption이 필요없음. 긴 길이의 long dependency에대해 대처 및 예측이 가능
  - ![image-20201113232823360](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20201113232823360.png)
  - Take a step of gradient descent to minimize negative log-likelihood.

- #### Loss Function of NNLM

  - Find theta that minimize negative log-likelihood
  - Find theta that minimize cross entropy with ground-truth probability distribution

- #### 요약

  - n-gram : traditional 한 ML 방식. 단어를 discrete symbol로 취급한다. 그래서 generalization issue가 발생하더라. 대신 빠른 연산과 쉽고 직관적임.
  - Neural Net LM: Word embedding을통해 unseen sequence에 대해 대처 가능하고 generalizion을 잘 한다. 단점은 feed forward 연산이 많다. 해석이 어려워진다.(XAI가 성립되기 어렵다..)

# Perplexity & Entropy

- #### Perplexity?

  - Sharp vs. Flat Distribution
  - ![image-20201113233931453](https://github.com/shoman2/shoman2.github.io/blob/master/assets/img/image-20201113233931453.png)
  - PPL이 높다? Flat한 분포. 즉 아무대서나 다 나타나더라.

- #### Information and Entropy

  - 정보이론에서 엔트로피는 어떤 정보의 불확실성을 나타낸다.
  - 불확실성은 일어날 것 같은 사건(likely event)의 확률
    - 자주 발생하는 사건은 낮은 정보량을 가지고
    - 드물게 발생하는 사건은 높은 정보량을 갖는다.
  - 불확실성 ) 1/확률 ) 정보량 
    - 즉, 확률에 반비례 해서 정보량이 있다.
    - 정보량
      - -log 때문에, 확률이 0에 가까워질수록 높은 정보량을 나타냄
      - I(x) = -logP(x)
    - 언어모델 관점에서 생각해보면 ?
      - 흔히 나올 수 없는 문장일수록 더 높은 정보량을 갖는다고 볼 수 있음.
  - Cross Entropy
    - ![image-20201113234538267](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20201113234538267.png)
    - ![image-20201113234722316](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20201113234722316.png)

- #### 요약

  - Objective: **Minimize PPL**
    - == **minimize cross entropy**
    - == **minimize negative log-likelihood**
  - 문장의 likelihood를 maximize하는 파라미터를 찾고 싶음
    - Ground-Truth 확률분포에 언어 모델을 근사하고 싶다.
  - GT분포와 LM분포 사이의 Cross Entropy를 구하고 Minimize 하는 것!
    - 문장의 PPL을 minimize
