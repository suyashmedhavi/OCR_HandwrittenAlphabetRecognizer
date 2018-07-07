# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:14:27 2018

@author: suyas
"""
from sklearn.metrics import auc,roc_curve
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from scipy import misc

model=load_model("C:/Users/suyas/Desktop/Project/Suyash/Model.h5")

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

def ROC(X_test,y_test,model):
    model_name = str(model).split('(')[0]
    
    fpr, tpr,_ = roc_curve( y_test,model.predict(X_test))
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    fig.canvas.set_window_title(model_name)
    
(x,y) = process_data("C:/Users/suyas/Desktop/Project/Suyash/Predict/")

img_x, img_y = 42, 32

x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

x = x.reshape(x.shape[0], img_x, img_y, 1)
x = x.astype('float32')
x /= 255

y = keras.utils.to_categorical(y, 27)

ROC(x,y,model)
