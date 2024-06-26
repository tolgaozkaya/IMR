import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL

# TPU (Tensor Processing Unit) kullanarak eğitim yapma denemesi
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU bul ve tanımla
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)  # TPU'ya bağlan
    tf.tpu.experimental.initialize_tpu_system(tpu)  # TPU sistemini başlat
    strategy = tf.distribute.experimental.TPUStrategy(tpu)  # TPU stratejisini belirle
# TPU bulunamazsa, varsayılan stratejiyi kullan
except:
    strategy = tf.distribute.get_strategy()  # Varsayılan strateji: CPU/GPU
print('Number of replicas:', strategy.num_replicas_in_sync)  # Kullanılacak replikaların sayısını yazdır

# TensorFlow sürümünü yazdır
print(tf.__version__)  # TensorFlow sürümünü yazdır

# Sabit değişkenler tanımla
AUTOTUNE = tf.data.experimental.AUTOTUNE  # Otomatik ayarlama değişkeni
BATCH_SIZE = 16 * strategy.num_replicas_in_sync  # Batch boyutu
IMAGE_SIZE = [176, 208]  # Görüntü boyutu
EPOCHS = 100  # Eğitim epoch sayısı

# Eğitim veri setini yükle
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# Doğrulama veri setini yükle
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# Sınıf isimlerini belirle
class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
train_ds.class_names = class_names  # Eğitim veri seti için sınıf isimlerini belirle
val_ds.class_names = class_names  # Doğrulama veri seti için sınıf isimlerini belirle
NUM_CLASSES = len(class_names)  # Sınıf sayısını belirle

# Verilerin görselleştirilmesi
plt.figure(figsize=(10, 10))  # Grafik boyutlarını ayarla
for images, labels in train_ds.take(1):  # Eğitim veri setinden bir batch al
  for i in range(9):  # İlk 9 görüntüyü görselleştir
    ax = plt.subplot(3, 3, i + 1)  # Alt grafik oluştur
    plt.imshow(images[i].numpy().astype("uint8"))  # Görüntüyü göster
    plt.title(train_ds.class_names[labels[i]])  # Başlık olarak sınıf ismini yaz
    plt.axis("off")  # Eksenleri kapat

# Etiketleri one-hot encoding'e dönüştürme
def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES)  # Etiketleri one-hot encoding yap
    return image, label  # Görüntü ve etiketi döndür

# Eğitim ve doğrulama veri setlerinde one-hot encoding uygulama
train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)  # Eğitim veri setine one-hot encoding uygula
val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)  # Doğrulama veri setine one-hot encoding uygula

# Verileri önbelleğe alma ve ön getirme
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)  # Eğitim veri setini önbelleğe al ve ön getirme yap
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  # Doğrulama veri setini önbelleğe al ve ön getirme yap

# Evrişim bloğu tanımlama
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),  # Ayrılabilir konvolüsyon katmanı
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),  # Ayrılabilir konvolüsyon katmanı
        tf.keras.layers.BatchNormalization(),  # Batch normalizasyon katmanı
        tf.keras.layers.MaxPool2D()  # Max pooling katmanı
    ])
    return block  # Konvolüsyon bloğunu döndür

# Yoğun katman bloğu tanımlama
def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),  # Yoğun katman
        tf.keras.layers.BatchNormalization(),  # Batch normalizasyon katmanı
        tf.keras.layers.Dropout(dropout_rate)  # Dropout katmanı
    ])
    return block  # Yoğun bloğu döndür

# Modeli oluşturma
def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMAGE_SIZE, 3)),  # Giriş katmanı
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),  # Konvolüsyon katmanı
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),  # Konvolüsyon katmanı
        tf.keras.layers.MaxPool2D(),  # Max pooling katmanı
        conv_block(32),  # 32 filtreli konvolüsyon bloğu
        conv_block(64),  # 64 filtreli konvolüsyon bloğu
        conv_block(128),  # 128 filtreli konvolüsyon bloğu
        tf.keras.layers.Dropout(0.2),  # Dropout katmanı
        conv_block(256),  # 256 filtreli konvolüsyon bloğu
        tf.keras.layers.Dropout(0.2),  # Dropout katmanı
        tf.keras.layers.Flatten(),  # Flatten katmanı
        dense_block(512, 0.7),  # 512 birimli yoğun blok
        dense_block(128, 0.5),  # 128 birimli yoğun blok
        dense_block(64, 0.3),  # 64 birimli yoğun blok
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Çıkış katmanı
    ])
    return model  # Modeli döndür

# Modeli strateji kapsamına al ve derle
with strategy.scope():
    model = build_model()  # Modeli oluştur
    METRICS = [tf.keras.metrics.AUC(name='auc')]  # Metrik olarak AUC kullan
    model.compile(
        optimizer='adam',  # Optimizasyon algoritması
        loss=tf.losses.CategoricalCrossentropy(),  # Kayıp fonksiyonu
        metrics=METRICS  # Metrikler
    )

# Öğrenme oranı için üstel azalma fonksiyonu tanımla
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)  # Üstel azalma fonksiyonu
    return exponential_decay_fn  # Üstel azalma fonksiyonunu döndür

# Geri çağırma fonksiyonlarını tanımla
exponential_decay_fn = exponential_decay(0.01, 20)  # Öğrenme oranı için üstel azalma
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)  # Öğrenme oranı zamanlayıcı
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("alzheimer_model.h5", save_best_only=True)  # Model kontrol noktası
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)  # Erken durdurma

# Modeli eğit
history = model.fit(
    train_ds,  # Eğitim veri seti
    validation_data=val_ds,  # Doğrulama veri seti
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],  # Geri çağırma fonksiyonları
    epochs=EPOCHS  # Epoch sayısı
)

# Modelin metriklerini görselleştir
fig, ax = plt.subplots(1, 2, figsize=(20, 3))  # Grafik oluştur
ax = ax.ravel()  # Alt grafikleri döndür
for i, met in enumerate(['auc', 'loss']):  # Her bir metrik için
    ax[i].plot(history.history[met])  # Eğitim verisi metriği
    ax[i].plot(history.history['val_' + met])  # Doğrulama verisi metriği
    ax[i].set_title('Model {}'.format(met))  # Başlık ayarla
    ax[i].set_xlabel('epochs')  # X ekseni etiketi
    ax[i].set_ylabel(met)  # Y ekseni etiketi
    ax[i].legend(['train', 'val'])  # Efsane ekle

# Test veri setini yükle ve değerlendirme yap
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)  # Test veri setine one-hot encoding uygula
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)  # Test veri setini önbelleğe al ve ön getirme yap
_ = model.evaluate(test_ds)  # Test veri seti ile modeli değerlendir
