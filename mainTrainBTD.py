import chardet
import warnings
import pandas as pd
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Activation, LeakyReLU, Dropout, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

import tensorflow as tf
import tensorflow_hub as hub
from efficientnet.keras import EfficientNetB0

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')

MAIN_DIR = "/Users/tolgaozkaya/Downloads/IMR/datasets/"
SEED = 40
os.listdir(MAIN_DIR)
for dirpath, dirnames, filenames in os.walk(MAIN_DIR):
    print(f"{len(dirnames)} directories and {len(filenames)} images in {dirpath}")

# Inspect the raw data before preprocessing


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

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
MAIN_DIR = "/Users/tolgaozkaya/Downloads/IMR/datasets/"

SEED = 40
IMG_SHAPE = (256, 256)
BATCH_SIZE = 32
INPUT_SIZE = 256

# Data generators
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

# Build the model
tf.random.set_seed(SEED)

model = Sequential()

# Load the EfficientNetB0 model pretrained on ImageNet
efficientnet = EfficientNetB0(include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
efficientnet.trainable = False

model.add(efficientnet)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(),
              metrics=["accuracy"])

# Fit the model
history = model.fit(train_data,
                    epochs=10,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))

# Save the model
model.save("EfficientNetB0.h5")
