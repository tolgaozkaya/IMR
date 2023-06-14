import os
from PIL import Image
import numpy as np
import smtplib
from email.message import EmailMessage

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

import joblib
from flask import Flask, render_template, request, jsonify, url_for
from skimage.io import imread
from keras.models import load_model

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Flask uygulamasını oluştur
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Dropout sınıfını özelleştirilmiş bir şekilde genişlet
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape if shape is not None else tf.shape(inputs)[i] for i, shape in enumerate(self.noise_shape)])

# Önceden eğitilmiş bir modeli yükle
with keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
    model = load_model('EfficientNetB7.h5')

# Beyin tumor tespiti yapmak için kullanılan endpoint
@app.route('/detect', methods=['POST'])
def detect():

    file = request.files['image']

    img = Image.open(file)
    img = img.convert("RGB")

    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array.reshape((1, 256, 256, 3))
    img_array = img_array / 255.0

    pred = model.predict(img_array)
    pred = np.round(pred)

    if pred[0][0] == 0:
        result_text = "Brain tumor not detected."
    else:
        result_text = "Brain tumor detected"

    return render_template('index.html', prediction=pred.tolist(), result=result_text)


svm = joblib.load('svm_model.joblib')

# Alzheimer tespiti yapmak için kullanılan endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Görüntüyü yükle ve ön işleme yap
    file = request.files['image']
    img = Image.open(file)
    img_size = (64, 64)

    # Görüntüyü img_size boyutuna yeniden boyutlandır
    img = img.resize(img_size)

    # Görüntüyü gri tonlamaya dönüştür
    img_gray = img.convert("L")

    # Görüntüyü numpy dizisine dönüştür ve düzleştir
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


@app.route('/send_mail', methods=['POST'])
def send_mail():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')

    if not name or not email or not subject or not message:
        return render_template('contact.html', error='Please fill in all fields.')

    msg = EmailMessage()
    msg.set_content(f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}")
    msg['Subject'] = subject
    msg['From'] = email
    msg['To'] = 'info@projectimr.com'

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('info@projectimr.com', 'your_password')
            server.send_message(msg)
        return render_template('contact.html', success=True)
    except smtplib.SMTPException:
        return render_template('contact.html', error='An error occurred while sending the email.')


@app.route('/stop', methods=['POST'])
def shutdown():
    if request.remote_addr != '127.0.0.1':
        return "Unauthorized", 403

    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_func()
    return 'Server shutting down...'


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


@app.route('/alzheimerdetection')
def alzheimerdetection():
    return render_template('alzheimerdetection.html')


if __name__ == '__main__':
    app.run(debug=True)
