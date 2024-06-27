import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import joblib

# Sabit değişkenler tanımla
IMAGE_SIZE = (176, 208)  # Görüntü boyutu
BATCH_SIZE = 32  # Batch boyutu

# Eğitim veri setini yükle
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    "./dataset_alzheimer",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training"
)
validation_generator = datagen.flow_from_directory(
    "./dataset_alzheimer",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)

# Sınıf isimlerini belirle
class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']

# Veri setini numpy dizilerine dönüştür
def generator_to_numpy(generator):
    images = []
    labels = []
    for i in range(len(generator)):
        batch_images, batch_labels = generator.next()
        for img, lbl in zip(batch_images, batch_labels):
            images.append(img)
            labels.append(lbl)
    return np.array(images), np.array(labels)

X_train, y_train = generator_to_numpy(train_generator)
X_val, y_val = generator_to_numpy(validation_generator)

# Verileri düzleştir
X_train_flatten = X_train.reshape((X_train.shape[0], -1))
X_val_flatten = X_val.reshape((X_val.shape[0], -1))

# SVM modelini oluştur ve eğit
svm_model = svm.SVC(kernel='linear', probability=True)
svm_model.fit(X_train_flatten, y_train)

# Modeli kaydet
joblib.dump(svm_model, 'svm_model.joblib')

# Modeli doğrulama verisi üzerinde değerlendir
y_val_pred = svm_model.predict(X_val_flatten)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Sınıflandırma raporu ve karışıklık matrisi
print("Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# Modeli eğitim verisi üzerinde değerlendir
y_train_pred = svm_model.predict(X_train_flatten)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Eğitim verisi için sınıflandırma raporu ve karışıklık matrisi
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred, target_names=class_names))

print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
