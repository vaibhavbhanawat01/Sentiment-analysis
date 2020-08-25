# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:44:48 2020

@author: Vaibhan
"""
# import keras libaries
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

baseModel = InceptionV3(weights = 'imagenet', include_top = False)
baseModel.summary()

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(units = 1024, activation = 'relu')(x)

prediction = Dense(units = 8, activation = 'softmax')(x)
model = Model(inputs = baseModel.input, outputs = prediction)
model.summary()

for i in baseModel.layers:
    i.trainable = False
dataset = pd.read_csv('dataset/train.csv')

train_image = []
for i in range(len(dataset)):
    img = image.load_img('dataset/train/'+dataset['Image'][i], target_size=(224,224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    train_image.append(img)
train_images = np.array(train_image)
train_category = dataset.iloc[:, 1]

labelEncoder = LabelEncoder()
train_category = labelEncoder.fit_transform(train_category)
onehotEncoder = OneHotEncoder()
train_reshape = np.reshape(train_category, (-1, 1));
train_category = onehotEncoder.fit_transform(train_reshape).toarray()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit(x = train_images, y = train_category, epochs=15, batch_size=5, verbose = 2)

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

model.summary()
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit(x = train_images, y = train_category, epochs=15, batch_size=5, verbose = 2)

dataset_test = pd.read_csv('dataset/test.csv')

# import the test images
prediction = []
for i in range(len(dataset_test)):
    img = image.load_img('dataset/test/'+dataset_test['Image'][i], 
                         target_size = (224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(np.array(img))
    category = np.argmax(pred)
    category = np.reshape(category, (-1, 1));
   
    prediction.append(labelEncoder.inverse_transform(category))
dataset_test['target'] = prediction
dataset_test.to_csv('dataset/test.csv', index = False)

# reading the actual predication
actual_predication = pd.read_excel('solution.xlsx')
actual = pd.DataFrame([sub.split(",") for sub in actual_predication['data']], columns=('Image', 'target'))
    
data_frame = pd.merge(dataset_test, actual, on='Image', how = 'inner')
correct = data_frame[data_frame['target_x'] == data_frame['target_y']]
print("number of correct predications", len(correct))
print("accuracy", len(correct)/ len(dataset_test))
    
