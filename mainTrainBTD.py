import os
import random
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import BinaryCrossentropy

from efficientnet.keras import EfficientNetB0

MAIN_DIR = "/Users/tolgaozkaya/Downloads/IMR/datasets/"
SEED = 40

# Rastgele bir görüntü göster
def view_random_image():
    subdirs = ['yes/', 'no/']
    subdir = np.random.choice(subdirs)
    target_folder = MAIN_DIR + subdir

    random_image = random.sample(os.listdir(target_folder), 1)

    img = cv2.imread(target_folder+random_image[0])
    plt.imshow(img, cmap="gray")
    plt.axis(False)
    plt.title(img.shape)
    plt.show()

# Sabitler
IMG_SHAPE = (256, 256)
BATCH_SIZE = 32
INPUT_SIZE = 256

# Veri üreteçleri
datagen = ImageDataGenerator(rescale=1/255., validation_split=0.5)

train_data = datagen.flow_from_directory(MAIN_DIR,
                                         target_size=IMG_SHAPE,
                                         batch_size=BATCH_SIZE,
                                         class_mode="binary",
                                         shuffle=True,
                                         subset="training")

test_data = datagen.flow_from_directory(MAIN_DIR,
                                        target_size=IMG_SHAPE,
                                        batch_size=BATCH_SIZE,
                                        class_mode="binary",
                                        shuffle=True,
                                        subset="validation")

# Modeli oluştur
model = Sequential()

efficientnet = EfficientNetB0(include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
efficientnet.trainable = False

model.add(efficientnet)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Modeli derle
model.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(),
              metrics=["accuracy"])

# Modeli eğit
history = model.fit(train_data,
                    epochs=10,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))

# Modeli kaydet
model.save("EfficientNetB0.h5")
