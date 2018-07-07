import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import os
from scipy import misc
import numpy as np

def process_data(path):
    x = []
    y = []
    for filename in os.listdir(path):
        temp = misc.imread(path + filename,flatten = True)
        print (filename)
        Letters = []
        Letters.append(temp[375:695,50:470])	        ####| Letter - A
        Letters.append(temp[375:695,600:1020])	     ####| Letter - B
        Letters.append(temp[375:695,1150:1570])	     ####| Letter - C
        Letters.append(temp[375:695,1700:2120])	     ####| Letter - D
        Letters.append(temp[765:1085,50:470])	     ####| Letter - E
        Letters.append(temp[765:1085,600:1020])	     ####| Letter - F
        Letters.append(temp[765:1085,1150:1570])	  ####| Letter - G
        Letters.append(temp[765:1085,1700:2120])	  ####| Letter - H
        Letters.append(temp[1155:1475,50:470])	     ####| Letter - I
        Letters.append(temp[1155:1475,600:1020])	  ####| Letter - J
        Letters.append(temp[1155:1475,1150:1570])	  ####| Letter - K
        Letters.append(temp[1155:1475,1700:2120])	  ####| Letter - L
        Letters.append(temp[1545:1865,50:470])	     ####| Letter - M
        Letters.append(temp[1545:1865,600:1020])	  ####| Letter - N
        Letters.append(temp[1545:1865,1150:1570])	  ####| Letter - O
        Letters.append(temp[1545:1865,1700:2120])	  ####| Letter - P
        Letters.append(temp[1935:2255,50:470])	     ####| Letter - Q
        Letters.append(temp[1935:2255,600:1020])	  ####| Letter - R
        Letters.append(temp[1935:2255,1150:1570])	  ####| Letter - S
        Letters.append(temp[1935:2255,1700:2120])	  ####| Letter - T
        Letters.append(temp[2325:2645,50:470])	     ####| Letter - U
        Letters.append(temp[2325:2645,600:1020])	  ####| Letter - V
        Letters.append(temp[2325:2645,1150:1570])	  ####| Letter - W
        Letters.append(temp[2325:2645,1700:2120])	  ####| Letter - X
        Letters.append(temp[2715:3035,50:470])	     ####| Letter - Y
        Letters.append(temp[2715:3035,600:1020])	  ####| Letter - Z
        
        c=1;
        for L in Letters:
            L = misc.imresize(L,(32,42))
            L = 255 - L
            L = L.flatten(order='C')
            x.append(L.ravel())
            y.append(c)
            c=c+1
    return x,y

batch_size = 80
num_classes = 27
epochs = 10

img_x, img_y = 42, 32

(x_train, y_train) = process_data("C:/Users/suyas/Desktop/Project/Suyash/Train/")
(x_test, y_test) = process_data("C:/Users/suyas/Desktop/Project/Suyash/Test/")

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

model.save("C:/Users/suyas/Desktop/Project/Suyash/Model.h5")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()