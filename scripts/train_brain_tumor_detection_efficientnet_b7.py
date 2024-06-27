import os
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import warnings
from keras.applications import EfficientNetB7

warnings.filterwarnings('ignore')  # Uyarıları devre dışı bırak

# Ana dizin ve rastgele tohum ayarı
MAIN_DIR = "./datasets/"
SEED = 40

# Görüntü veri üretici ayarları
IMG_SHAPE = (224, 224)  # Görüntü boyutu
BATCH_SIZE = 32  # Batch boyutu

datagen = ImageDataGenerator(rescale=1/255., validation_split=0.5)  # Veri jeneratörü
train_data = datagen.flow_from_directory(MAIN_DIR, target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, subset="training")  # Eğitim verileri
test_data = datagen.flow_from_directory(MAIN_DIR, target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, subset="validation")  # Test verileri

# EfficientNetB7 ile Transfer Öğrenme
effnet_model = Sequential([
    EfficientNetB7(include_top=False, weights="imagenet", input_shape=IMG_SHAPE + (3,)),  # EfficientNetB7 katmanı
    GlobalAveragePooling2D(),  # Global ortalama havuzlama
    Dense(1, activation="sigmoid")  # Çıkış katmanı
])

# EfficientNetB7 modelinin derlenmesi
effnet_model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=["accuracy"])  # Modeli derle

# EfficientNetB7 modelinin eğitilmesi
effnet_history = effnet_model.fit(train_data, epochs=10, steps_per_epoch=len(train_data), validation_data=test_data, validation_steps=len(test_data))  # Modeli eğit

# EfficientNetB7 modelinin kaydedilmesi
effnet_model.save("efficientnet_b7_model.h5")  # Modeli kaydet

# Eğitim ve doğrulama eğrilerinin çizilmesi
def plot_curves(history):
    loss = history.history["loss"]  # Eğitim kayıp değerleri
    val_loss = history.history["val_loss"]  # Doğrulama kayıp değerleri
    accuracy = history.history["accuracy"]  # Eğitim doğruluk değerleri
    val_accuracy = history.history["val_accuracy"]  # Doğrulama doğruluk değerleri
    epochs = range(len(history.history["loss"]))  # Epoch sayısı

    plt.plot(epochs, loss, label="training_loss")  # Eğitim kayıp eğrisi
    plt.plot(epochs, val_loss, label="val_loss")  # Doğrulama kayıp eğrisi
    plt.title("Kayıp")  # Başlık
    plt.xlabel("Epochs")  # X ekseni etiketi
    plt.legend()  # Efsane
    plt.show()  # Grafiği göster

    plt.plot(epochs, accuracy, label="training_accuracy")  # Eğitim doğruluk eğrisi
    plt.plot(epochs, val_accuracy, label="val_accuracy")  # Doğrulama doğruluk eğrisi
    plt.title("Doğruluk")  # Başlık
    plt.xlabel("Epochs")  # X ekseni etiketi
    plt.legend()  # Efsane
    plt.show()  # Grafiği göster

# EfficientNetB7 modeli için eğitim eğrilerinin çizilmesi
plot_curves(effnet_history)

# EfficientNetB7 modelinin değerlendirilmesi
effnet_result = effnet_model.evaluate(test_data, verbose=0)  # Modeli değerlendir
print(f"Değerlendirme Doğruluğu: {effnet_result[1]*100:.2f}%\nKayıp: {effnet_result[0]:.4f}")  # Değerlendirme sonuçlarını yazdır
