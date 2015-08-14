# Colours

- 11,271 images in total
- 1,228 images showing blue flowering plants (0)
- 778 images showing brown flowering plants (1)
- 1,352 images showing green flowering plants (2)
- 1,227 images showing orange flowering plants (3)
- 2,400 images showing pink flowering plants (4)
- 633 images showing purple flowering plants (5)
- 510 images showing red flowering plants (6)
- 1,865 images showing white flowering plants (7)
- 1,219 images showing yellow flowering plants (8)

## Configuration

- Number for training: 10,144
- Number for validation: 1,127

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
('Epoch ', 6)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 1.5425 - train accuracy:: 0.4463
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 7)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 1.4453 - train accuracy:: 0.4827
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 8)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 1.2962 - train accuracy:: 0.5349
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 9)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 1.1316 - train accuracy:: 0.5920
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 10)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.9721 - train accuracy:: 0.6533
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 11)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.8007 - train accuracy:: 0.7190
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 12)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.6783 - train accuracy:: 0.7651
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 13)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.5529 - train accuracy:: 0.8029
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 14)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.4674 - train accuracy:: 0.8350
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 15)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.4082 - train accuracy:: 0.8570
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 16)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.3367 - train accuracy:: 0.8809
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 17)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.3020 - train accuracy:: 0.8944
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 18)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.2773 - train accuracy:: 0.9036
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 19)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.2490 - train accuracy:: 0.9137
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 20)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.2437 - train accuracy:: 0.9155
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 21)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.2099 - train accuracy:: 0.9280
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 22)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.1963 - train accuracy:: 0.9345
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 23)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.1862 - train accuracy:: 0.9366
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 24)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.1655 - train accuracy:: 0.9446
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 25)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.1534 - train accuracy:: 0.9455
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 26)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.1540 - train accuracy:: 0.9463
Running a little validation...
1127/1127 [==============================] - 1s
('Epoch ', 27)
Training...
10100/10144 [============================>.] - ETA: 0s - train loss: 0.1394 - train accuracy:: 0.9497
Running a little validation...
1127/1127 [==============================] - 1s
```
