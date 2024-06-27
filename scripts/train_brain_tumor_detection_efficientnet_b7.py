import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import warnings
import random
import cv2

warnings.filterwarnings('ignore')  # Uyarıları devre dışı bırak

# Ana dizin ve rastgele tohum ayarı
MAIN_DIR = "./datasets/"
SEED = 40

# Görüntü veri üretici ayarları
IMG_SHAPE = (224, 224)  # Görüntü boyutu
BATCH_SIZE = 32  # Batch boyutu

# Veri jeneratörü
datagen = ImageDataGenerator(rescale=1/255., validation_split=0.5)
train_data = datagen.flow_from_directory(MAIN_DIR, target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, subset="training")
test_data = datagen.flow_from_directory(MAIN_DIR, target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, subset="validation")

# EfficientNetB7 ile Transfer Öğrenme
effnet_url = "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1"
effnet_layer = hub.KerasLayer(effnet_url, trainable=False, name="feature_extraction_layer")

effnet_model = Sequential([
    effnet_layer,
    Dense(1, activation='sigmoid')
])

effnet_model.build([None, IMG_SHAPE[0], IMG_SHAPE[1], 3])

# Modelin derlenmesi
effnet_model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=["accuracy"])

# Modelin eğitilmesi
effnet_history = effnet_model.fit(train_data, epochs=10, steps_per_epoch=len(train_data), validation_data=test_data, validation_steps=len(test_data))

# Modelin kaydedilmesi
effnet_model.save("models/efficientnet_b7_model.h5")

# Eğitim ve doğrulama eğrilerinin çizilmesi
def plot_curves(history):
    epochs = range(len(history.history["loss"]))
    
    plt.figure(figsize=(14, 5))
    
    # Kayıp eğrisi
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], label="Eğitim Kaybı")
    plt.plot(epochs, history.history["val_loss"], label="Doğrulama Kaybı")
    plt.title("Eğitim ve Doğrulama Kaybı")
    plt.xlabel("Epochs")
    plt.ylabel("Kayıp")
    plt.legend()
    
    # Doğruluk eğrisi
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], label="Eğitim Doğruluğu")
    plt.plot(epochs, history.history["val_accuracy"], label="Doğrulama Doğruluğu")
    plt.title("Eğitim ve Doğrulama Doğruluğu")
    plt.xlabel("Epochs")
    plt.ylabel("Doğruluk")
    plt.legend()
    
    plt.show()

# Eğitim eğrilerini çiz
plot_curves(effnet_history)

# Modelin değerlendirilmesi
effnet_result = effnet_model.evaluate(test_data, verbose=0)
print(f"Değerlendirme Doğruluğu: {effnet_result[1]*100:.2f}%\nKayıp: {effnet_result[0]:.4f}")
