---
layout: post
title:  "Keras - CNN Model Code Test"
subtitle:   "CNN"
categories: data
tags: dl
comments: true
---
```python
import keras
keras.__version__

```




    '2.4.3'




```python
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

```


```python
model.summary()

```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
    =================================================================
    Total params: 55,744
    Trainable params: 55,744
    Non-trainable params: 0
    _________________________________________________________________



```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

```


```python
model.summary()

```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                36928     
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 93,322
    Trainable params: 93,322
    Non-trainable params: 0
    _________________________________________________________________



```python
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

```


```python
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

```

    Epoch 1/5
    938/938 [==============================] - 34s 36ms/step - loss: 0.1794 - accuracy: 0.9444
    Epoch 2/5
    938/938 [==============================] - 34s 37ms/step - loss: 0.0477 - accuracy: 0.9857
    Epoch 3/5
    938/938 [==============================] - 33s 36ms/step - loss: 0.0326 - accuracy: 0.9898
    Epoch 4/5
    938/938 [==============================] - 34s 36ms/step - loss: 0.0249 - accuracy: 0.9924
    Epoch 5/5
    938/938 [==============================] - 25s 27ms/step - loss: 0.0206 - accuracy: 0.9936





    <tensorflow.python.keras.callbacks.History at 0x7fec1079a190>




```python
test_loss, test_acc = model.evaluate(test_images, test_labels)

```

    313/313 [==============================] - 1s 5ms/step - loss: 0.0351 - accuracy: 0.9888



```python
test_acc

```




    0.9887999892234802




```python

```
