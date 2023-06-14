import os
import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import BinaryCrossentropy

import tensorflow as tf
from efficientnet.keras import EfficientNetB7

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')

# Paths
PROJECT_FOLDER = "/Users/tolgaozkaya/Downloads/IMR/"
MAIN_DIR = os.path.join(PROJECT_FOLDER, "datasets")

# Sabitler
SEED = 40
IMG_SHAPE = (256, 256)
BATCH_SIZE = 32
INPUT_SIZE = 256

os.listdir(MAIN_DIR)
for dirpath, dirnames, filenames in os.walk(MAIN_DIR):
    print(f"{len(dirnames)} dizin ve {len(filenames)} görüntü {dirpath} içinde bulunuyor")

# Veri oluşturucular
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

# Modeli oluşturma
tf.random.set_seed(SEED)

model = Sequential()

# ImageNet'te önceden eğitilmiş EfficientNetB7 modelini yükleme
efficientnet = EfficientNetB7(
    include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
efficientnet.trainable = False

model.add(efficientnet)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(),
              metrics=["accuracy"])

# Modeli eğitme
history = model.fit(train_data,
                    epochs=10,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))

# Modeli kaydetme
model.save(os.path.join(PROJECT_FOLDER, "EfficientNetB7.h5"))
