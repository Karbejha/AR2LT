# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 01:35:14 2022

@author: hp
"""
from __future__ import division, print_function, absolute_import

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random

import cv2
from tensorflow import keras

image=cv2.imread('images/letters.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,5)
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 127, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
letters=thresh.reshape([-1, 32, 32, 1])
print(letters.shape)


model = keras.models.load_model('D:\python\OCR tutorial\model')
batch_size=64
learning_rate = 0.01
epochs=50
num_classes=28
trainx = pd.read_csv("D:/python/OCR tutorial/Dataset/Arabic Handwritten Characters Dataset CSV/csvTrainImages 13440x1024.csv",header=None)
trainy = pd.read_csv("D:/python/OCR tutorial/Dataset/Arabic Handwritten Characters Dataset CSV/csvTrainLabel 13440x1.csv",header=None)

testx = pd.read_csv("D:/python/OCR tutorial/Dataset/Arabic Handwritten Characters Dataset CSV/csvTestImages 3360x1024.csv",header=None)
testy = pd.read_csv("D:/python/OCR tutorial/Dataset/Arabic Handwritten Characters Dataset CSV/csvTestLabel 3360x1.csv",header=None)




chars =list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
chars 
X_train = []
y_train=list(trainy[0])
for i in range(len(trainx)):
    img = trainx.iloc[i,:].values.reshape(32,32,1)
    X_train.append(img)
X_test = []
y_test=list(testy[0])
for i in range(len(testx)):
    img = testx.iloc[i,:].values.reshape(32,32,1)
    X_test.append(img)
def display_images(X_train,row,col):
    fig = plt.figure(figsize=(12,12))
    ax = fig.subplots(row,col)
    for j in range(row):
        for i in range(col):
            fig.suptitle('Images')
            ax[j,i].imshow(X_train[random.randint(0,len(X_train))])
    plt.plot()
display_images(X_train,5,5)
display_images(X_test,3,3)
def preprocess_data(train_data_x):
    train_data_x = train_data_x.to_numpy().reshape((train_data_x.shape[0], 32, 32)).astype('uint8')
    for i in range(len(train_data_x)):
        train_data_x[i] = cv2.rotate(train_data_x[i], cv2.ROTATE_90_CLOCKWISE)      # Rotating the images.
        train_data_x[i] = np.flip(train_data_x[i], 1)                               # Flipping the images
    train_data_x = train_data_x.reshape([-1, 32, 32, 1]).astype('uint8')          # Reshaping into the required size.
    train_data_x = train_data_x.astype('float32')/255                             # Here we normalize our images.
    return np.asarray(train_data_x)
X_train =preprocess_data(trainx)
display_images(X_train,5,5)
X_test =preprocess_data(testx)
display_images(X_test,5,5)
classes=np.unique(y_train)
plt.pie(trainy[0].value_counts(), labels=classes, colors=['#90EE91', '#F47174'], autopct='%1.1f')
plt.show()

y_pred = model.predict(image)
print(classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1),target_names=chars))
def display_images_result(X_train,row,col):
    fig = plt.figure(figsize=(12,12))
    ax = fig.subplots(row,col)
    for j in range(row):
        for i in range(col):
            fig.suptitle('Images')
            idx=random.randint(0,len(X_train))
            ax[j,i].imshow(X_train[idx])
            pred=model.predict(np.expand_dims(X_test[idx], axis=0))
            ax[j,i].set_title(chars[(np.argmax(pred,axis=1)[0])],fontdict={'fontsize':20,})
    plt.plot()

display_images_result(X_test,10,10)


