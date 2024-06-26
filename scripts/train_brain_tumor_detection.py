# Gerekli kütüphanelerin içe aktarılması
# Bu kod parçası, model eğitimi, veri işleme, görüntüleme ve uyarıların devre dışı bırakılması için gerekli kütüphaneleri içe aktarır.
import pandas as pd
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow uyarılarını devre dışı bırak

from keras.models import Sequential, load_model  # Model oluşturma ve yükleme
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D  # Katmanlar
from keras.losses import BinaryCrossentropy  # Kayıp fonksiyonu
from keras.optimizers import Adam  # Optimizasyon algoritması
from keras.preprocessing.image import ImageDataGenerator  # Görüntü veri jeneratörü
from sklearn.metrics import classification_report, confusion_matrix  # Değerlendirme metrikleri
import warnings

warnings.filterwarnings('ignore')  # Uyarıları devre dışı bırak

# Ana dizin ve rastgele tohum ayarı
# Ana dizin, görüntülerin bulunduğu yerdir ve SEED, modelin tekrarlanabilir sonuçlar üretmesi için kullanılan rastgele tohum değeridir.
MAIN_DIR = "/Users/tolgaozkaya/IMR/datasets/"
SEED = 40

# Veri kümesinden rastgele bir görüntü görüntüleme fonksiyonu
# Bu fonksiyon, veri kümesinden rastgele bir görüntüyü seçip gösterir.
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
# Yukarıdaki fonksiyonu kullanarak veri kümesinden rastgele bir görüntü gösterilir.
view_random_image()

# Görüntü veri üretici ayarları
# ImageDataGenerator ile veri kümesinin eğitim ve test için bölünmesi ve yeniden ölçeklendirilmesi sağlanır.
IMG_SHAPE = (224, 224)  # Görüntü boyutu
BATCH_SIZE = 32  # Batch boyutu

datagen = ImageDataGenerator(rescale=1/255., validation_split=0.5)  # Veri jeneratörü
train_data = datagen.flow_from_directory(MAIN_DIR, target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, subset="training")  # Eğitim verileri
test_data = datagen.flow_from_directory(MAIN_DIR, target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, subset="validation")  # Test verileri

# CNN modelinin oluşturulması ve eğitilmesi
# Basit bir Convolutional Neural Network (CNN) modeli oluşturulur ve eğitilir.
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
# Model, Binary Crossentropy kayıp fonksiyonu ve Adam optimizer kullanılarak derlenir.
cnn_model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=["accuracy"])  # Modeli derle

# CNN modelinin eğitilmesi
# Model, eğitim verileri kullanılarak 10 epoch boyunca eğitilir ve doğrulama verileri ile değerlendirilir.
cnn_history = cnn_model.fit(train_data, epochs=10, steps_per_epoch=len(train_data), validation_data=test_data, validation_steps=len(test_data))  # Modeli eğit

# CNN modelinin kaydedilmesi
# Eğitim tamamlandıktan sonra model dosyaya kaydedilir.
cnn_model.save("models/cnn_model.h5")  # Modeli kaydet

# Eğitim ve doğrulama eğrilerinin çizilmesi
# Bu fonksiyon, eğitim ve doğrulama sırasında kaydedilen kayıp ve doğruluk değerlerini çizerek görselleştirir.
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
# Yukarıdaki fonksiyon kullanılarak CNN modelinin eğitim eğrileri çizilir.
plot_curves(cnn_history)

# CNN modelinin değerlendirilmesi
# Eğitim sonrası model test verileri ile değerlendirilir ve doğruluk ile kayıp değerleri ekrana yazdırılır.
cnn_result = cnn_model.evaluate(test_data, verbose=0)  # Modeli değerlendir
print(f"Değerlendirme Doğruluğu: {cnn_result[1]*100:.2f}%\nKayıp: {cnn_result[0]:.4f}")  # Değerlendirme sonuçlarını yazdır

# EfficientNetB7 ile Transfer Öğrenme
# EfficientNetB7 modelinin önceden eğitilmiş ağırlıkları kullanılarak transfer öğrenme uygulanır.
effnet_url = "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1"  # EfficientNetB7 URL'si
effnet_layer = hub.KerasLayer(effnet_url, trainable=False, name="feature_extraction_layer")  # EfficientNetB7 katmanı

effnet_model = Sequential([  # Model oluştur
    effnet_layer,  # EfficientNetB7 katmanı
    Dense(1, activation="sigmoid")  # Çıkış katmanı
])

# EfficientNetB7 modelinin derlenmesi
# Model, Binary Crossentropy kayıp fonksiyonu ve Adam optimizer kullanılarak derlenir.
effnet_model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=["accuracy"])  # Modeli derle

# EfficientNetB7 modelinin eğitilmesi
# Model, eğitim verileri kullanılarak 10 epoch boyunca eğitilir ve doğrulama verileri ile değerlendirilir.
effnet_history = effnet_model.fit(train_data, epochs=10, steps_per_epoch=len(train_data), validation_data=test_data, validation_steps=len(test_data))  # Modeli eğit

# EfficientNetB7 modelinin kaydedilmesi
# Eğitim tamamlandıktan sonra model dosyaya kaydedilir.
effnet_model.save("models/effnetB7_model.h5")  # Modeli kaydet

# EfficientNetB7 modeli için eğitim eğrilerinin çizilmesi
# EfficientNetB7 modelinin eğitim eğrileri yukarıdaki plot_curves fonksiyonu kullanılarak çizilir.
plot_curves(effnet_history)

# EfficientNetB7 modelinin değerlendirilmesi
# Eğitim sonrası EfficientNetB7 modeli test verileri ile değerlendirilir ve doğruluk ile kayıp değerleri ekrana yazdırılır.
effnet_result = effnet_model.evaluate(test_data, verbose=0)  # Modeli değerlendir
print(f"Değerlendirme Doğruluğu: {effnet_result[1]*100:.2f}%\nKayıp: {effnet_result[0]:.4f}")  # Değerlendirme sonuçlarını yazdır
