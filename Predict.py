from keras.models import load_model
import os
from scipy import misc
import numpy as np
import keras

model=load_model("C:/Users/suyas/Desktop/Project/Suyash/Model.h5")

num_classes = 27

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

(x,y) = process_data("C:/Users/suyas/Desktop/Project/Suyash/Predict/")

img_x, img_y = 42, 32

x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

x = x.reshape(x.shape[0], img_x, img_y, 1)
x = x.astype('float32')
x /= 255

y = keras.utils.to_categorical(y, num_classes)

y_pre = model.predict(x,verbose=1)

letters = y_pre.argmax(axis=1)

acc = letters == y.argmax(axis=1)
a=sum(letters == y.argmax(axis=1))
print (a)
print (2600-a)
print ("ACCURACY: ", sum(acc)/len(acc))

answer=[]
for i in letters:
    answer.append(chr(i+64))
    
k=0.9311538461538461+0.9534615384615385+0.9584615384615385+0.9315384615384615+0.9046153846153846
k/=5
