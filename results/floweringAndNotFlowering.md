# Flowering and not Flowering

- 16,498 images in total
- 8,249 images showing flowering plants
- 8,249 images showing plants of which aren't flowering

## Configuration

- Number for training: 14,850
- Number for validation: 1,648

```python2
model = Sequential()

model.add(Convolution2D(64, 3, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(65536, 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, nb_classes))
model.add(Activation('softmax'))

rms = Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=rms)

```

## Results

```
Using gpu device 0: Quadro K2200
X_train shape: (14850, 3, 64, 64)
Y_train shape: (14850, 2)
train samples: 14850 validation samples: 1648

Epoch  0
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.6655 - train accuracy:: 0.6597
Valid
1648/1648 [==============================] - 6s

Epoch  1
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.5076 - train accuracy:: 0.7493
Valid
1648/1648 [==============================] - 6s

Epoch  2
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.4636 - train accuracy:: 0.7788
Valid
1648/1648 [==============================] - 6s

Epoch  3
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.4314 - train accuracy:: 0.7961
Valid
1648/1648 [==============================] - 6s

Epoch  4
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.4052 - train accuracy:: 0.8139
Valid
1648/1648 [==============================] - 6s

Epoch  5
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.3764 - train accuracy:: 0.8318
Valid
1648/1648 [==============================] - 6s

Epoch  6
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.3362 - train accuracy:: 0.8502
Valid
1648/1648 [==============================] - 5s

Epoch  7
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.3024 - train accuracy:: 0.8694
Valid
1648/1648 [==============================] - 5s

Epoch  8
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.2528 - train accuracy:: 0.8923
Valid
1648/1648 [==============================] - 5s

Epoch  9
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.2214 - train accuracy:: 0.9103
Valid
1648/1648 [==============================] - 5s

Epoch  10
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.1786 - train accuracy:: 0.9331
Valid
1648/1648 [==============================] - 6s

Epoch  11
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.1419 - train accuracy:: 0.9447
Valid
1648/1648 [==============================] - 5s

Epoch  12
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.1160 - train accuracy:: 0.9572
Valid
1648/1648 [==============================] - 6s

Epoch  13
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.1191 - train accuracy:: 0.9556
Valid
1648/1648 [==============================] - 5s

Epoch  14
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0880 - train accuracy:: 0.9678
Valid
1648/1648 [==============================] - 5s

Epoch  15
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0762 - train accuracy:: 0.9725
Valid
1648/1648 [==============================] - 5s

Epoch  16
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0631 - train accuracy:: 0.9778
Valid
1648/1648 [==============================] - 5s

Epoch  17
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0602 - train accuracy:: 0.9776
Valid
1648/1648 [==============================] - 5s

Epoch  18
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0522 - train accuracy:: 0.9823
Valid
1648/1648 [==============================] - 5s

Epoch  19
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0445 - train accuracy:: 0.9844
Valid
1648/1648 [==============================] - 5s

Epoch  20
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0450 - train accuracy:: 0.9852
Valid
1648/1648 [==============================] - 5s

Epoch  21
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0380 - train accuracy:: 0.9864
Valid
1648/1648 [==============================] - 5s

Epoch  22
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0339 - train accuracy:: 0.9877
Valid
1648/1648 [==============================] - 5s

Epoch  23
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0346 - train accuracy:: 0.9878
Valid
1648/1648 [==============================] - 6s

Epoch  24
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0364 - train accuracy:: 0.9866
Valid
1648/1648 [==============================] - 6s

Epoch  25
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0321 - train accuracy:: 0.9881
Valid
1648/1648 [==============================] - 6s

Epoch  26
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0301 - train accuracy:: 0.9901
Valid
1648/1648 [==============================] - 5s

Epoch  27
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0253 - train accuracy:: 0.9907
Valid
1648/1648 [==============================] - 5s

Epoch  28
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0293 - train accuracy:: 0.9907
Valid
1648/1648 [==============================] - 5s
```
