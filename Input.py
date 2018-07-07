import os
from scipy import misc
import numpy as np
import pickle

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

(x_train, y_train) = process_data("D:/Project/Suyash/Train/")
(x_test, y_test) = process_data("D:/Project/Suyash/Test/")

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

pickle.dump(x_train,open("D:/Project/Suyash/x_train.obj","wb"))
pickle.dump(y_train,open("D:/Project/Suyash/y_train.obj","wb"))
pickle.dump(x_test,open("D:/Project/Suyash/x_test.obj","wb"))
pickle.dump(y_test,open("D:/Project/Suyash/y_test.obj","wb"))