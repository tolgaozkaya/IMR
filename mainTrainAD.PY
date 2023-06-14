import os
from glob import glob
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.io import imread
from skimage.transform import resize
import joblib

# Alzheimer veri seti klasörlerinin yolları
very_mild = glob(r'/Users/tolgaozkaya/Downloads/IMR/alzheimerdataset/Very_Mild_Demented/*.jpg')
mild = glob(r'/Users/tolgaozkaya/Downloads/IMR/alzheimerdataset/Mild_Demented/*.jpg')
moderate = glob(r'/Users/tolgaozkaya/Downloads/IMR/alzheimerdataset/Moderate_Demented/*.jpg')

# Alzheimer olmayan veri seti klasörünün yolu
non = glob(r'/Users/tolgaozkaya/Downloads/IMR/alzheimerdataset/Non_Demented/*.jpg')

# Alzheimer veri seti için görüntüleri ve etiketleri yükleme
X = []
y = []

for img_path in very_mild:
    img = imread(img_path)
    img_resized = resize(img, (64, 64))  # Görüntüleri (64, 64) boyutuna yeniden boyutlandırma
    X.append(img_resized)
    y.append(0)

for img_path in mild:
    img = imread(img_path)
    img_resized = resize(img, (64, 64))
    X.append(img_resized)
    y.append(1)

for img_path in moderate:
    img = imread(img_path)
    img_resized = resize(img, (64, 64))
    X.append(img_resized)
    y.append(2)

# Alzheimer olmayan veri seti için görüntüleri ve etiketleri yükleme
for img_path in non:
    img = imread(img_path)
    img_resized = resize(img, (64, 64))
    X.append(img_resized)
    y.append(3)

# Veriyi eğitim ve test setlerine bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi Numpy dizilerine dönüştürme
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Veriyi SVM giriş gereksinimlerine uyacak şekilde yeniden şekillendirme
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Doğrusal çekirdekli SVM ile eğitim
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)

# SVM'i test etme ve doğruluk hesaplama
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

joblib.dump(svm, 'svm_model.joblib')
