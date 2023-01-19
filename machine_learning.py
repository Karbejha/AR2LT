from __future__ import division, print_function, absolute_import

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, \
    Flatten
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from PIL import Image
import cv2

batch_size = 64
learning_rate = 0.01
epochs = 50
num_classes = 28
trainx = pd.read_csv(
    "D:/python/OCR tutorial/Dataset/Arabic Handwritten Characters Dataset CSV/csvTrainImages 13440x1024.csv",
    header=None)
trainy = pd.read_csv(
    "D:/python/OCR tutorial/Dataset/Arabic Handwritten Characters Dataset CSV/csvTrainLabel 13440x1.csv", header=None)

testx = pd.read_csv(
    "D:/python/OCR tutorial/Dataset/Arabic Handwritten Characters Dataset CSV/csvTestImages 3360x1024.csv", header=None)
testy = pd.read_csv("D:/python/OCR tutorial/Dataset/Arabic Handwritten Characters Dataset CSV/csvTestLabel 3360x1.csv",
                    header=None)
image = cv2.imread('images/letters.jpg')
chars = list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
chars
# Latin=list('alef':u'\u0623','ba':u'\u0628','ta':'ت','tha':'ث','gim':'ج','ha':'ح','ka':'خ','dal':'د','thal':'ذ','ra':'ر','zyn':'ز','sin':'س','shin':'ش','sad':'ص','taa':'ط','thaa':'ظ','ain':'ع','ghayn':'غ','fa':'ف','qaf':'ق','kaf':'ك','lam':'ل','mim':'م','nun':'ن','ha':'ه','waw':'و','ya':'ي')
# Latin

X_train = []
y_train = list(trainy[0])
for i in range(len(trainx)):
    img = trainx.iloc[i, :].values.reshape(32, 32, 1)
    X_train.append(img)
X_test = []
y_test = list(testy[0])
for i in range(len(testx)):
    img = testx.iloc[i, :].values.reshape(32, 32, 1)
    X_test.append(img)


def display_images(X_train, row, col):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots(row, col)
    for j in range(row):
        for i in range(col):
            fig.suptitle('Images')
            ax[j, i].imshow(X_train[random.randint(0, len(X_train))])
    plt.plot()


display_images(X_train, 5, 5)
display_images(X_test, 3, 3)


def preprocess_data(train_data_x):
    train_data_x = train_data_x.to_numpy().reshape((train_data_x.shape[0], 32, 32)).astype('uint8')
    for i in range(len(train_data_x)):
        train_data_x[i] = cv2.rotate(train_data_x[i], cv2.ROTATE_90_CLOCKWISE)  # Rotating the images.
        train_data_x[i] = np.flip(train_data_x[i], 1)  # Flipping the images
    train_data_x = train_data_x.reshape([-1, 32, 32, 1]).astype('uint8')  # Reshaping into the required size.
    train_data_x = train_data_x.astype('float32') / 255  # Here we normalize our images.
    return np.asarray(train_data_x)


X_train = preprocess_data(trainx)
display_images(X_train, 5, 5)
X_test = preprocess_data(testx)
display_images(X_test, 5, 5)
classes = np.unique(y_train)
plt.pie(trainy[0].value_counts(), labels=classes, colors=['#90EE91', '#F47174'], autopct='%1.1f')
plt.show()


def create_model(activation='relu', optimizer='adam', kernel_initializer='he_normal'):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 1), activation=activation,
                     kernel_initializer=kernel_initializer))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation=activation, kernel_initializer=kernel_initializer))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same', activation=activation, kernel_initializer=kernel_initializer))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(32, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(28, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()  # Now we created an instance of a model with our custom architefture.
model.summary()
model = create_model(optimizer=tf.keras.optimizers.Adamax(0.001),
                     kernel_initializer='uniform',
                     activation='relu')  # Then we display our model's summary.
y_train = tf.keras.utils.to_categorical(np.array(y_train) - 1  # Returns an array of dimentions (13340, 28).
                                        , num_classes=28)
y_test = tf.keras.utils.to_categorical(np.array(y_test) - 1  # Returns an array of dimentions (3360, 28).
                                       , num_classes=28)
history = model.fit(X_train,
                    y_train,
                    validation_split=0.3,
                    epochs=30,
                    batch_size=64, )
y_pred = model.predict(image)
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=chars))
model.save('D:\python\OCR tutorial\model')


def display_images_result(X_train, row, col):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots(row, col)
    for j in range(row):
        for i in range(col):
            fig.suptitle('Images')
            idx = random.randint(0, len(X_train))
            ax[j, i].imshow(X_train[idx])
            pred = model.predict(np.expand_dims(X_test[idx], axis=0))
            ax[j, i].set_title(chars[(np.argmax(pred, axis=1)[0])], fontdict={'fontsize': 20, })
    plt.plot()


from matplotlib import rcParams

display_images_result(X_test, 10, 10)
