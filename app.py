from skimage.io import imread
import joblib
from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from PIL import Image
import tensorflow_hub as hub
import smtplib
from keras.models import load_model
from keras.utils import custom_object_scope
from ultralytics import YOLO

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Flask uygulamasını oluştur
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape if shape is not None else tf.shape(inputs)[i] for i, shape in enumerate(self.noise_shape)])


# EffnetB7 modelini özel nesne kapsamı ile yükle
with custom_object_scope({'FixedDropout': FixedDropout, 'KerasLayer': hub.KerasLayer}):
    effnet_model = load_model('models/efficientnet_b0_model.h5')

# YOLOv8 modelini yükle
yolo_model = YOLO('models/best_model.pt')

# SVM modelini yükle
svm = joblib.load('models/svm_model.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/service1')
def service1():
    return render_template('service1.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/blogs')
def blogs():
    return render_template('blogs.html')


@app.route('/blog1')
def blog1():
    return render_template('blog1.html')


@app.route('/blog2')
def blog2():
    return render_template('blog2.html')


@app.route('/blog3')
def blog3():
    return render_template('blog3.html')


@app.route('/alzheimerdetection')
def alzheimerdetection():
    return render_template('alzheimerdetection.html')


@app.route('/braintumordetectwithyolo')
def braintumordetectwithyolo():
    return render_template('braintumordetectwithyolo.html')

# Beyin tümörü tespiti için EffecientNetB7 modelini kullanır
@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']

    img = Image.open(file)
    img = img.convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array.reshape((1, 256, 256, 3))
    img_array = img_array / 255.0

    pred = effnet_model.predict(img_array)
    pred = np.round(pred)

    if pred[0][0] == 0:
        result_text = "Brain tumor not detected."
    else:
        result_text = "Brain tumor detected"

    return render_template('index.html', prediction=pred.tolist(), result=result_text)

# Alzheimer tespiti için SVM modelini kullanır
@app.route('/predict', methods=['POST'])
def predict():
    # Görüntüyü yükle ve ön işlem yap
    file = request.files['image']
    img = Image.open(file)
    img_size = (64, 64)

    # Görüntüyü img_size boyutuna yeniden boyutlandır
    img = img.resize(img_size)

    # Görüntüyü gri tonlamaya çevir
    img_gray = img.convert("L")

    # Görüntüyü numpy dizisine çevir ve düzleştir
    img_gray_arr = np.array(img_gray)
    img_gray_arr_flat = img_gray_arr.reshape(1, -1)

    # Piksel değerlerini 0 ile 1 arasında normalize et
    img_gray_arr_flat_norm = img_gray_arr_flat / 255.0
    img_processed = img_gray_arr_flat_norm

    # SVM modeli ile tahmin yap
    prediction = svm.predict(img_processed)[0]

    # Tahmin için sınıf adını al
    class_names = ["Very_Mild_Demented", "Mild_Demented",
                   "Moderate_Demented", "Non_Demented"]
    class_name = class_names[prediction]
    result = "Level Of : " + class_name

    return render_template('alzheimerdetection.html', image=img, prediction=prediction.tolist(), result=result)

# YOLO modeli kullanarak nesne tespiti yapar
@app.route('/yolodetect', methods=['POST'])
def yolodetect():
    file = request.files['image']

    img = Image.open(file)
    img_path = os.path.join('uploads', file.filename)
    img.save(img_path)

    # Görüntüyü yükle ve modelle tespit yap
    results = yolo_model(img_path)

    # Benzersiz bir dosya adı oluştur
    result_img_name = f'result_{file.filename}'

    # Tespit sonuçlarını işleyin ve görselleştirin
    for i, result in enumerate(results):
        result_img = result.plot()  # Görselleştirilmiş görüntüyü al
        result_img = Image.fromarray(result_img.astype(np.uint8))  # numpy array'den PIL Image'e dönüştür
        result_img_path = os.path.join('static/predictions', result_img_name)
        result_img.save(result_img_path)  # Görselleştirilmiş görüntüyü kaydet

    return render_template('braintumordetectwithyolo.html', filename=result_img_name)

# E-posta gönderme işlevi
@app.route('/send_mail', methods=['GET', 'POST'])
def send_mail():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # email body
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"

        # email gönder
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login('your_email@gmail.com', 'your_password')
            server.sendmail('your_email@gmail.com',
                            'tolgaozkya14@gmail.com', body)
            server.quit()
            return render_template('contact.html', success=True)
        except:
            return render_template('contact.html', success=False)

    return render_template('contact.html', success=None)

# Sunucuyu kapatma işlevi
@app.route('/stop', methods=['POST'])
def shutdown():
    # İsteğin güvenilir bir kaynaktan geldiğini kontrol et
    if request.remote_addr != '127.0.0.1':
        return "Unauthorized", 403

    # Sunucuyu kapat
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_func()
    return 'Server shutting down...'

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static/predictions'):
        os.makedirs('static/predictions')
    app.run(debug=True)
