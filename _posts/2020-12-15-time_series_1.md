---
layout: post
title:  "Time Series 데이터 분석 - 1"
subtitle:   "System Trading"
categories: data
tags: dl
comments: true
---

## Time Series
시계열, 타임시리즈 데이터에 분석에 대해 알아보자.
![](https://i2.wp.com/radacad.com/wp-content/uploads/2017/07/trendseasonal.png)

본 포스팅 에서는 **시계열 인코딩과 리커런트 네트워크**를 논한다. 본  두 가지 주제는 시간이 지남에 따라 발생하는 데이터를 처리하는 방법이기 때문에 합쳐서 바라보기 좋다. 

시계열 데이터 인코딩은 뉴럴넷에 시간이 지남에 따라 발생하는 이벤트를 보여준다. 시간이 지남에 따라 뉴럴넷에 발생하는 데이터를 인코딩하는 많은 다양한 방법이 존재한다. 

인코딩이 필요한 이유는 피드포워드 네트웍이 항상 주어진 입력 벡터에 대해 동일한 출력 벡터를 생성하기 때문이다. 리커런트 뉴럴넷은 시간에 따라 발생하는 데이터를 자동으로 처리할 수 있기 때문에 시계열 데이터의 인코딩을 요구하지 않는다.



우리가 일상에서 마주치는 날씨 온도의 변화는 시계열 데이터의 흔한 예일 것이다. 예를 들어, 만약 우리가 오늘의 온도가 섭씨 25도이고 내일의 온도가 섭씨 27도라는 것을 안다면, 리커런트 네트웍과 시계열 인코딩은 한 주의 정확한 온도를 예측하기 위한 또 다른 방법을 제공해준다고 볼 수 있다. 

반대로, 전통적인 피드포워드 뉴럴넷은 항상 주어진 입력에 대해 동일한 출력으로 나타낼 것이다. 만약 우리가 내일의 온도를 예측하기 위해 피드포워드 신경망을 트레이닝 시킨다면, 25도에 대해 27도의 값을 반환해야 한다. 25도가 주어지면 항상 27도가 출력된다는 사실이 예측함에 있어서 방해가 될 수 있다. 27도의 기온이 항상 25도를 따라서 나타나는 것은 아니다. 

예측 시점 전의 일련의 온도를 고려하는 것이 더 나을 것이다. 아마도 지난 주의 기온은 우리가 내일의 기온을 어느정도 예측할 수 있게 해 줄 것이다. 따라서, 반복 리커런트 뉴럴넷과 시계열 인코딩은 신경망에 대한 시간에 따른 데이터를 나타내는 두 가지 다른 접근 방식을 나타낸다.

## LSTM의 이해

지금까지 우리가 봐온 신경망은 항상 피드포워드 형태 였다.이 유형의 신경망은 항상 첫 번째 은닉 계층에 연결된 입력 계층으로 시작한다. 각 히든 레이어는 항상 다음 히든레이어에 연결되어 있다. 그리고 가장 마지막 히든레이어는 출력층으로 연결된다. 이렇기 때문에 본 네트워크를 형태를 "피드 포워드" 네트워크라고 한다.

순환 신경 네트워크는 후방 연결도 허용되기 때문에 그렇게 견고하지 않다. 반복적 연결은 한 층의 뉴런을 이전 층이나 뉴런 그 자체와 연결시킨다. 대부분의 반복 신경 네트워크 아키텍처는 반복 연결에서 상태를 유지한다. 

피드포워드 신경망은 어떤 상태도 유지하지 않는다. 반복된 신경 네트워크의 상태는 신경 네트워크의 일종의 단기 기억으로 작용한다. 결과적으로, 반복 신경 네트워크는 항상 주어진 입력에 대해 동일한 출력을 생성하지는 않을 것이다.

반복 신경 네트워크는 연결이 한 계층에서 다음 계층으로, 입력 계층에서 출력 계층으로만 흐르도록 강요하지 않는다. 반복 연결은 뉴런과 다음 다른 유형의 뉴런 중 하나 사이에 연결이 형성될 때 발생합니다.

- 뉴런 그 자체

- 동일한 레벨의 뉴런

- 이전 레벨의 뉴런

반복 연결은 입력 뉴런이나 편향 뉴런을 목표로 삼을 수 없다.
반복 연결을 처리하는 것은 어렵다. 순환 링크는 끝없는 루프를 만들기 때문에, 신경 네트워크는 언제 멈출지 알 수 있는 어떤 방법을 가지고 있어야 한다. 끝없는 루프에 진입한 신경망은 유용하지 않을 것이다. 무한 루프를 방지하기 위해 다음 세 가지 방법으로 반복 연결을 계산할 수 있다.

- 맥락 뉴런
- 고정된 반복 횟수에 대한 출력 계산
- 뉴런 출력이 안정화될 때까지 출력 계산

문맥 뉴런은 특별한 뉴런 유형으로, 입력 내용을 기억하고 다음에 네트워크를 계산할 때 입력값을 출력으로 제공한다. 예를 들어, 컨텍스트 뉴런 0.5를 입력으로 제공하면 0이 출력된다. 컨텍스트 뉴런은 항상 첫 번째 호출 시 0을 출력한다. 그러나 컨텍스트 뉴런을 입력으로 0.6을 제공하면 출력은 0.5가 된다. 우리는 문맥 뉴런에 대한 입력 연결의 가중치를 측정 할 수 없지만, 다른 네트워크 연결과 마찬가지로 문맥 뉴런의 출력의 무게를 잴 수 있다.

문맥 뉴런은 우리가 하나의 피드포워드 패스로 신경망을 계산할 수 있게 해준다. 문맥 뉴런은 보통 층에서 발생한다.



```python
%matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def f2(x):
    a = []
    for item in x:
        a.append(math.tanh(item))
    return a

x = np.arange(-10., 10., 0.2)
y1 = sigmoid(x)
y2 = f2(x)

print("Sigmoid")
plt.plot(x,y1)
plt.show()

print("Hyperbolic Tangent(tanh)")
plt.plot(x,y2)
plt.show()

```

**LSTM의 구조 1**![](https://camo.githubusercontent.com/833a6504b300950efc764d52dabb10d232491404/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6a656666686561746f6e2f7438315f3535385f646565705f6c6561726e696e672f6d61737465722f696d616765732f636c6173735f31305f6c73746d312e706e67)

**LSTM의 구조 2**![](https://camo.githubusercontent.com/17bb4659628de279abae38f783f2555c399f1590/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6a656666686561746f6e2f7438315f3535385f646565705f6c6561726e696e672f6d61737465722f696d616765732f636c6173735f31305f6c73746d322e706e67)

## Simple LSTM Example (TF) Code

```python
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
import numpy as np

max_features = 4 # 0,1,2,3 (total of 4)
x = [
    [[0],[1],[1],[0],[0],[0]],
    [[0],[0],[0],[2],[2],[0]],
    [[0],[0],[0],[0],[3],[3]],
    [[0],[2],[2],[0],[0],[0]],
    [[0],[0],[3],[3],[0],[0]],
    [[0],[0],[0],[0],[1],[1]]
]
x = np.array(x,dtype=np.float32)
y = np.array([1,2,3,2,3,1],dtype=np.int32)

# Convert y2 to dummy variables
y2 = np.zeros((y.shape[0], max_features),dtype=np.float32)
y2[np.arange(y.shape[0]), y] = 1.0
print(y2)

print('Build model...')
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, \
               input_shape=(None, 1)))
model.add(Dense(4, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x,y2,epochs=200)
pred = model.predict(x)
predict_classes = np.argmax(pred,axis=1)
print("Predicted classes: {}",predict_classes)
print("Expected classes: {}",predict_classes)

```

**추가 읽어보면 좋은 자료 :**

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

 