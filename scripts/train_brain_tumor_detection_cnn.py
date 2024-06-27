import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings('ignore')  # Uyarıları devre dışı bırak

# Ana dizin ve rastgele tohum ayarı
MAIN_DIR = "./datasets/"
SEED = 40

# Veri kümesinden rastgele bir görüntü görüntüleme fonksiyonu
def view_random_image():
    subdirs = ['yes/', 'no/']  # Alt dizinler
    subdir = np.random.choice(subdirs)  # Rastgele bir alt dizin seç
    target_folder = MAIN_DIR + subdir  # Hedef klasör yolu
    random_image = random.sample(os.listdir(target_folder), 1)  # Rastgele bir görüntü seç
    img = cv2.imread(target_folder + random_image[0])  # Görüntüyü oku
    plt.imshow(img, cmap="gray")  # Görüntüyü gri ölçekle göster
    plt.axis(False)  # Eksenleri kapat
    plt.title(img.shape)  # Görüntü boyutunu başlık olarak ayarla
    plt.show()  # Görüntüyü göster

# Rastgele bir görüntü görüntüleme
view_random_image()

# Görüntü veri üretici ayarları
IMG_SHAPE = (224, 224)  # Görüntü boyutu
BATCH_SIZE = 32  # Batch boyutu

datagen = ImageDataGenerator(rescale=1/255., validation_split=0.5)  # Veri jeneratörü
train_data = datagen.flow_from_directory(MAIN_DIR, target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, subset="training")  # Eğitim verileri
test_data = datagen.flow_from_directory(MAIN_DIR, target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, subset="validation")  # Test verileri

# CNN modelinin oluşturulması ve eğitilmesi
tf.random.set_seed(SEED)  # Rastgele tohum ayarla

cnn_model = Sequential([  # Model oluştur
    Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3)),  # Konvolüsyon katmanı
    Conv2D(32, 3, activation='relu'),  # Konvolüsyon katmanı
    MaxPool2D(pool_size=2),  # Max pooling katmanı
    Conv2D(32, 3, activation='relu'),  # Konvolüsyon katmanı
    Conv2D(16, 3, activation='relu'),  # Konvolüsyon katmanı
    MaxPool2D(2, padding='same'),  # Max pooling katmanı
    Flatten(),  # Flatten katmanı
    Dense(1, activation='sigmoid')  # Çıkış katmanı
])

# CNN modelinin derlenmesi
cnn_model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=["accuracy"])  # Modeli derle

# CNN modelinin eğitilmesi
cnn_history = cnn_model.fit(train_data, epochs=10, steps_per_epoch=len(train_data), validation_data=test_data, validation_steps=len(test_data))  # Modeli eğit

# CNN modelinin kaydedilmesi
cnn_model.save("models/cnn_model.h5")  # Modeli kaydet

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

# CNN modeli için eğitim eğrilerinin çizilmesi
plot_curves(cnn_history)

# CNN modelinin değerlendirilmesi
cnn_result = cnn_model.evaluate(test_data, verbose=0)  # Modeli değerlendir
print(f"Değerlendirme Doğruluğu: {cnn_result[1]*100:.2f}%\nKayıp: {cnn_result[0]:.4f}")  # Değerlendirme sonuçlarını yazdır
