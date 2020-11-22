---
layout: post
title:  "자연어처리 - 인트로덕션 (LM)"
subtitle:   "Language Modeling"
categories: data
tags: dl
comments: true
---
# Intro to Language Modeling(LM)

- 언어모델, LM은 "문장의 확률"을 나타낸 모델
  - 구체적으로는 '문장 자체의 출현 확률'을 예측하는 모델
  - 또는 이전 단어들이 주어졌을 때 '다음 단어'를 예측하기 위한 모델
- 우리 머릿속에는 단어와 단어 사이의 확률이 우리도 모르게 학습되어 있다.
- 많은 문장들을 수집하여, 단어와 단어 사이의 출현 빈도를 세어 확률을 계산!
- 궁극적인 목표는 우리가 일상 생활에서 사용하는 언어의 문장 분포를 정확하게 모델링 하는 것/ 또는 잘 근사(Approximation) 하는 것
  - 특정 도메인의 문장의 분포를 파악하기 위해서 해당 분야의 말뭉치 Corpus를 수집하기도 한다. (어른과 어린이의 LM이 다르고,, 의사와 일반인이 다르고.. 등등)
- 한국어 NLP는 왜 어렵나? 바로 교착어이기 때매.. 어순이 안중요. 접사에 따라 역할이 정해지기 떄문.. 단어와 단어 사이의 확률을 계산하는데 불리하게 작용하는.. 그리고 생략도 가능하기 때매 종종..
- 따라서, 확률이 퍼지는 현상이 한국말엔 존재하게 됨
- 접사를 따로 분리해주지 않으면 어휘의 수가 기하급수적으로 늘어나 희소성이 더욱 늘어난다.
- 언어모델 LM의 적용분야 (NLG Task에대해 매우 중요한 역할을 하더라..)
  - 1)Speech Recognition: Acoustic Model과 결합하여, 인식된 Phone의 sequence에 대해서 좀 더 높은 확률을 갖는 sequence로 보완
  - 2)번역 모델과 결합하여, 번역 된 결과 문장을 자유스럽게 만듦
  - 3)OCR : 인식된 character candidate sequence에 대해서 좀 더 높은 확률을 갖는 sequence를 선택하도록 도움
  - 4)Other NLG Tasks: 뉴스기사 생성, 챗봇, 검색어 자동완성 등등.
- ASR(Automatic Speech Recognition)
  - x=음성, y=word sequence
  - argmaxP(x|y)P(y) === AM과 LM의 확률의 곱임.

# Language Modeling(LM)

- Objective: x~P(x), Sampling 하여 모은 데이터 셋은 D라고 놓는다.

- Theta^ = argmaxSIGMA i=1 ~ N, logP(x i:n; Theta)

- Chain Rule (backpropagation 의 chain rule이랑은 좀 다름)

  - We can convert joint prob to conditional prob.

  - P(A,B,C,D) = P(D|A,B,C)P(A,B,C)

    ​                   =P(D|A,B,C)P(C|A,B)P(A,B).... P(x1:n)

  - BOS : Beginning of Sentence

  - EOS: End of Sentence

- USING LM.. 

  - Pick better fluent sentence ()
  - Predict next word given previous words (NLG)

- 요약

  - 언어모델은 주어진 코퍼스 문장들의 likelihood를 최대화 하는 파라미터 theta를 찾아내, 주어진 코퍼스를 기반으로 언어의 분포를 학습한다.
    - 즉, 코퍼스 기반으로 문장들에 대한 확률 분포 함수를 근사한다.
  - 문장의 확률은 단어가 주어졌을 때, 다음 단어를 예측하는 확률을 차례대로 곱한 것과 같다.
  - 따라서 언어모델링은 주어진 단어가 있을 때, 다음 단어의 likelihood를 최대화 하는 파라미터 theta를 찾는 과정이라고도 볼 수 있다.
    - 주어진 단어들이 있을 때, 다음 단어에 대한 확률 분포 함수를 근사하는 과정

  ​                    

# n-gram Language Model

What is good model?

- Generalization

  - Training(seen) data를 통해 test(unseen) data에 대해 훌륭한 prediction을 할 수 있는가?
  - 만약 모든 경우의 수에 대해 학습 데이터를 모을 수 있다면, table look-up으로 모든 문제를 풀 수 있을 것이다. 그러나 이 자체가 불가능한게 현실이기 때문에 generalization 능력이 중요

- Count based approximation - 모두 관련된 문장들 끌고와서 카운트한다.

  - Given Sentence, we can approximate conditional probability by counting word sequence.
  - 그러나..  그런 문장 순서가 존재하지 않는다면..? 어쩔것인가? What if there is no such word sequence?

- Apply Markov Assumption

  - Approximate with counting only previous k tokens.
  - 결국 앞에나온 k개의 단어만 보고 판단하겠다는 가정
  - 이걸 word보다 sentence레벨로 확장해본다면, we can cover more word sequences, even if they are unseen in training corpus

- n-gram

  - n = k+1
  - k=0, 1-gram = uni-gram
  - k=1, 2-gram = bi-gram
  - k=2, 3-gram = tri-gram 이라고 부른다.

-  n-gram에서  n이 커질수록 오히려 확률이 정확하게 표현되는데 어렵다. 따라서 적절한 n을 사용하자

- 보통 tri-gram을 가장 많이 사용한다.

- corpus(말뭉치)의 양이 많을 때는 4-gram을 사용하기도 한다. 언어모델의 성능은 크게 오르지 않는데 반해, 단어 조합의 경우의 수가 너무 크게 증가하므로 효율이 떨어짐

- How to Train/Inference n-gram LM?

  - SRILM - 다운로드 받아서 쓸 수 있음..참고 바람

- 요약

  - 확률값을 근사하는 가장 간단한 방법은 코퍼스에서 빈도를 카운트 하는 것
    - 그러나 복잡한 문장일수록 코퍼스에서 출현 빈도가 낮아, 부정확한 근사가 이루어질 것
  - 따라서, Markov assumption을 도입하여 확률값을 근사하자
    - 학습 코퍼스에서 보지 못한 문장에 대해서도 확률값을 구할 수 있다.
    - n의 크기가 중요하고 tri-gram을 많이 사용하고 적당하다..

  

# Smoothing and Discounting

n-gram의 성능을 개선하기 위한 테크닉

Smoothing

- Markov assumption을 도입하였지만 여전히 문제는 남아있다. Training corpus에 없는 unseen word sequence의 확률은 0인가??

- Unseen word sequence에 대한 대처로 Smoothing or Discounting이 존재

- Popular algorithm

  - Modified Kneser-Ney Discounting (KN Discounting)

- 스무딩 테크닉1: Add one Smoothing

  -  to prevent count becomes zero. 1을 더해버려서 분자가 0이되는 걸 방지. 분모는 vocab size 를 더해서 undefined되는 걸 방지

- 스무딩 테크닉2: Kneser-Nay Discounting

  - 다양한 단어 뒤에서 나타나는 단어일수록 unseen word sequence에 등장 할 확률이 높지 않을까?
    - 앞에 등장한 단어의 종류가 다양할수록 해당 확률이 높을 것 같음.

- 요약

  - Markove Assumption

    - Count기반 Approximation
    - 긴 word seq는 학습 코퍼스에 존재하지 않을 수 있다. Markov assumption을 통해 근거리의 단어만 고려됨

  - Smoothing and Discounting

    - Markov assumption을 통해서도 여전히 확률값이 0이 될 수 있음
    - Smoothing 또는 Discounting을 통해 현상을 완화!
    - 여전히 unseen word sequence에 대한 대처는 미흡

    

