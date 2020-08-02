# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:19:22 2020

@author: vaibhav_bhanawat
"""



# import keras libaries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical


# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(128, 128, 3), 
                             activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Convolution 2nd layer
classifier.add(Convolution2D(128, 3, 3, border_mode='same', activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

## Full connection- making classic ANN
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 8, activation = 'softmax'))

# Compiling the Model
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

dataset = pd.read_csv('dataset/train.csv')

train_image = []
for i in range(len(dataset)):
    img = image.load_img('dataset/train/'+dataset['Image'][i], target_size=(128,128,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
train_images = np.array(train_image)
train_category = dataset.iloc[:, 1]

labelEncoder = LabelEncoder()
train_category = labelEncoder.fit_transform(train_category)
onehotEncoder = OneHotEncoder()
train_category = onehotEncoder.fit_transform(train_category).toarray()


train_category = to_categorical(train_category, 8)


classifier.fit(train_images, train_category, epochs=15, batch_size=64)

dataset_test = pd.read_csv('dataset/test.csv')

# import the test images

prediction = []
for i in range(len(dataset_test)):
    img = image.load_img('dataset/test/'+dataset_test['Image'][i], 
                         target_size = (128, 128, 3))
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    pred = classifier.predict_classes(np.array(img))
    prediction.append(labelEncoder.inverse_transform(pred))
dataset_test['target'] = prediction
dataset_test.to_csv('dataset/test.csv', index = False)
    
