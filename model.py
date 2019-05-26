import os
import csv

samples = []
data_path = "../STD/data/driving_log.csv"
with open(data_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]  

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import math

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            current_path = '../STD/data/IMG/' # fill in the path to your training IMG directory
            for batch_sample in batch_samples:
                name_0 = current_path +batch_sample[0].split('/')[-1]
                name_1 = current_path +batch_sample[1].split('/')[-1]
                name_2 = current_path +batch_sample[2].split('/')[-1]
                # print('name_0:',name_0)
                # print('name_1:',name_1)
                center_image = cv2.imread(name_0)
                # print(center_image)
                center_angle = float(batch_sample[3].strip())
                images.append(center_image)
                angles.append(center_angle)

                center_image_flipped = cv2.flip(center_image, 1)
                center_angle_flipped = center_angle*-1.0
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)
                
                correction = 0.2 # this is a parameter to tune

                left_image = cv2.imread(name_1)
                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)

                right_image = cv2.imread(name_2)
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format
    

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

activation_relu = 'relu'
model = Sequential()
# model.add(Flatten(input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(row,col, ch)))
model.add(Lambda(lambda x: x/255.0 - 0.5)) # pixel_mean_centered

model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='same'))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='same'))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same'))
model.add(Activation(activation_relu))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())
# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))
model.add(Dense(100))
model.add(Activation(activation_relu))
model.add(Dense(50))
model.add(Activation(activation_relu))
model.add(Dense(10))
model.add(Activation(activation_relu))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)
model.save('model_v6_ge.h5')

import matplotlib.pyplot as plt
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
plt.savefig('result')