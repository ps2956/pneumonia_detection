import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
import pandas as pd
import numpy as np

train_idg = ImageDataGenerator(
    rescale = 1./255.,
    rotation_range = 30,  
    zoom_range = 0.2, 
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip = True,
)
test_idg = ImageDataGenerator(
    rescale = 1./255.
)
val_idg = ImageDataGenerator(
    rescale=1./255.
)

IMG_SIZE = (224, 224)

train = train_idg.flow_from_directory('chest x-rays/chest_xray/train', 
                                      class_mode='binary',
                                      color_mode="grayscale",
                                      target_size = IMG_SIZE
                                     )
test = test_idg.flow_from_directory('chest x-rays/chest_xray/test', 
                                    class_mode='binary',
                                    color_mode="grayscale",
                                    target_size = IMG_SIZE
                                   )

trainX, trainY = next(train)
testX, testY = next(test)
print(len(trainX), len(trainY))

print('Train X=%s Y=%s' %(trainX.shape, trainY.shape))
print('Test X=%s Y=%s' %(testX.shape, testY.shape))

labels = ['Pneumonia' if label == 0 else 'Normal' for label in trainY]
sns.countplot(labels)

for idx in range(10):
    plt.figure(figsize=(5,5))
    plt.imshow(trainX[idx].reshape(224,224), cmap='gray')
    plt.title(labels[idx])

print("x_train shape:", trainX.shape)
print(trainX.shape[0], "train samples")
print(testX .shape[0], "test samples")

model = Sequential(name = "Sequential")
model.add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (224, 224,1) , name="FirstConvolution"))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same' , name="FirstMaxPool"))
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , name="SecondConvolution"))
model.add(Dropout(0.1, name="Droupout1"))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same' , name="SecondMaxPool"))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , name="ThirdConvolution"))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same' , name="ThirdMaxPool"))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , name="ForthConvolution"))
model.add(Dropout(0.2 , name="Droupout2"))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same' , name="ForthMaxPool"))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , name="FifthConvolution"))
model.add(Dropout(0.2 , name="Droupout3"))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same' , name="FifthMaxPool"))
model.add(Flatten(name="Flatten"))
model.add(Dense(units = 128 , activation = 'relu' , name="Dense1"))
model.add(Dense(units = 64 , activation = 'relu' , name="Dense2"))
model.add(Dropout(0.2 , name="Droupout4"))
model.add(Dense(units = 32 , activation = 'relu' , name="Dense3"))
model.add(Dense(units = 1 , activation = 'sigmoid' , name="Dense4"))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

history = model.fit(train, epochs = 20 , callbacks = [learning_rate_reduction])

print("Loss of the model is - " , model.evaluate(testX,testY)[0])
print("Accuracy of the model is - " , model.evaluate(testX,testY)[1]*100 , "%")

predictions = model.predict_classes(testX)
predictions = predictions.reshape(1,-1)[0]
print(predictions[:15])

print(classification_report(testY, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

print('Confusion Matrix\n')
matrix = confusion_matrix(testY,predictions)
print(matrix)

correct = np.nonzero(predictions == testY)[0]
incorrect = np.nonzero(predictions != testY)[0]
print(len(correct), len(incorrect))

i = 0
for c in correct[:6]:
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(testX[c].reshape(224, 224), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], testY[c]))
    plt.show()
    i += 1

i = 0
for c in incorrect[:6]:
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(testX[c].reshape(224, 224), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], testY[c]))
    plt.show()
    i += 1