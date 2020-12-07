---
layout: post
title:  "자연어처리 - Interpolation + PPL "
subtitle:   "Language Modeling"
categories: data
tags: dl
comments: true
---
# Interpolation & Back-off
- 수치 보간법이라고 한글로 불리움.. 수학과 수업에서 들어본 듯.
- 다른 LM을 linear하게 일정 비율로 섞는 것
- general domain LM + domain specific LM = general domain에서 잘 동작하는 domatin adapted LM
- 예시: 의료/법률/특허 관련 AST/MT  system등이 있다.
- 추가 질문
  - 그냥 domain specific corpus로 LM을 만들면 안되는지?
    - 그렇게 되면 unseen word seq가 너무 많을 것 같다..
  - 그냥 전체 corpus를 합쳐서 LM을 만들면 안되나요?
    - Domain specific corpus의 양이 너무 적어서 반영이 안될 수도?
  - Interpolation에서 ratio(lambda)를 조절하여 중요(weight)를 조절
    - 명시적으로 (explicit) 섞을 수 있다.
    - General domain test set, Domain specific test set 모두에서 좋은 성능을 찾는 hyper-parameter Lambda를 찾아야 한다.
- Back-off (뒤로 가면서 n을 줄여가는 것)
  - 희소성에 대처하는 방법
  - Markov assumption처럼 n을 점점 줄여가면 ?
    - 조건부 확률에서 조건부 word seq를 줄여가면, unknown word가 없다면 언젠가는 확률을 구할 수 있다!

- 요약
  - Back-off를 통해 확률값이 0이 되는 현상은 방지가능 -OoV 제외
    - 하지만 unseen word sequence를 위해 back-off를 거치는 순간 확률 값이 매우 낮아져 버림. 여전히 음성인식등의 활용에서 어려움이 남음
  - 전통적인 NLP에서는 단어를 discrete symbol로 보기 때문에 문제
    - exact matching에 대해서만 count를 하며, 확률 값을 approximation
    - 다양한 방법을 통해 문제를 완화하려는 하지만 근본적인 해결책 x
      - Markov Assumption
      - Smoothing and Discounting
      - Interpolation and Back-off

# 언어모델 LM의 평가방식 - Perplexity

- Intrinsic evaluation (정성평가)
  - 정확도 / 시간과 비용이 많이듬 / 사람이 직접 체크한다고 보면됨
- Extrinsic evaluation(정량평가)
  - 시간과 비용을 아낄 수 있음 / 컴퓨터가 자동으로 계산한다고 보면 됨
  - 정량평가는 정성평가와 비슷할수록 좋은 방법!
- 다시한번 생각해 보자. What is Good LM ?
  - 실제 사용하는 언어의 분포를 가장 잘 근사한 모델
    - 실제사용하는 언어? == 테스트 시 입력 문장들
    - 분포를 잘 근사한다? == 문장의 likelihood가 높을 것
  - 잘 정의된 테스트셋의 문장에 대해서 높은 확률을 반환하는 언어모델이 좋은 모델!
- Evaluation 
  - Perplexity(PPL - 줄여서 이렇게 부름)
    - 테스트 문장에 대해서 언어모델을 이용하여 확률(likelihood)을 구하고
    - PPL 수식에 넣어 언어모델의 성능 측정
      - 문장의 확률을 길이에 대해서 normalization (기하평균)
      - ![image-20201113225814313](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20201113225814313.png)
      - 테스트 문장에 대해서 확률을 높게 반환할수록 좋은 LM이다.
      - 테스트 문장에 대한 PPL이 작을수록 좋은 LM 이다.
- 요약
  - 좋은 언어모델: 잘 정의된 테스트셋 문장에 대해서 높은 확률 (낮은 PPL)을 갖는 모델
  - PPL: 작을수록 좋다. 확률의 역수에 문장 길이로 기하 평균한 값. 
    - every time-step 마다 평균적으로 헷갈리고 있는 단어의 수라고 정의 가능

