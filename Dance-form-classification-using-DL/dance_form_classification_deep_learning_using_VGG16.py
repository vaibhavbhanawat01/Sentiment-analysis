# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:19:22 2020

@author: vaibhav_bhanawat
"""
# import keras libaries
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import shutil
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = pd.read_csv('dataset/train.csv')

os.chdir('dataset/train/')
classes = []
for i in range(len(dataset)):
    name = dataset['target'][i]
    fileName = dataset['Image'][i]
    if os.path.isdir(name) is False:
        classes.append(name)
        os.mkdir(name)
    shutil.move(fileName, name)
os.chdir('../../')

train_generator = ImageDataGenerator(rotation_range  = 10, width_shift_range = 0.1, height_shift_range=0.1, \
                    shear_range=0.1, zoom_range=0.1,channel_shift_range=10,horizontal_flip = True,
                    preprocessing_function = preprocess_input) \
                    .flow_from_directory('dataset/train', target_size = (224,224), batch_size = 10)

baseModel = VGG16(weights = 'imagenet', include_top = True)
model = Sequential()
for i in baseModel.layers[:-1]:
    model.add(i)

model.add(Dense(units = 8, activation = 'softmax'))

for i in model.layers[:-6]:
    i.trainable = False
model.summary()

model.compile(optimizer = Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x = train_generator, epochs=15, verbose = 2)

dataset_test = pd.read_csv('dataset/test.csv')
labelEncoder = LabelEncoder()
labelEncoder.fit(classes)

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
    
