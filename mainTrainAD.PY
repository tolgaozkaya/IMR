from glob import glob
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Paths to Alzheimer dataset folders
very_mild = glob(r'/Users/tolgaozkaya/Downloads/IMR/alzheimerdataset/Very_Mild_Demented/*.jpg')
mild = glob(r'/Users/tolgaozkaya/Downloads/IMR/alzheimerdataset/Mild_Demented/*.jpg')
moderate = glob(r'/Users/tolgaozkaya/Downloads/IMR/alzheimerdataset/Moderate_Demented/*.jpg')

# Paths to non-Alzheimer dataset folder
non = glob(r'/Users/tolgaozkaya/Downloads/IMR/alzheimerdataset/Non_Demented/*.jpg')

# Load images and labels for Alzheimer dataset
X = []
y = []

for img_path in very_mild:
    img = imread(img_path)
    img_resized = resize(img, (64, 64)) # Resize image to (64, 64) for faster training
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

# Load images and labels for non-Alzheimer dataset
for img_path in non:
    img = imread(img_path)
    img_resized = resize(img, (64, 64))
    X.append(img_resized)
    y.append(3)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape data to fit SVM input requirements
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train SVM with linear kernel
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)

# Test SVM and compute accuracy
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(svm, 'svm_model.joblib')