from skimage.io import imread
import joblib
from flask import Flask, render_template, request
import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow_hub as hub
import smtplib
from keras.models import load_model
from flask import url_for
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Create a Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape if shape is not None else tf.shape(inputs)[i] for i, shape in enumerate(self.noise_shape)])


# Load the model with custom object scope
with keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
    model = load_model(
        '/Users/tolgaozkaya/Downloads/IMR/EfficientNetB0.h5')


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


@app.route('/predict', methods=['POST'])
def predict():
    # Load image and preprocess
    file = request.files['image']
    img = Image.open(file)
    img_size = (64, 64)

    # Resize image to img_size
    img = img.resize(img_size)

    # Convert image to grayscale
    img_gray = img.convert("L")

    # Convert image to numpy array and flatten
    img_gray_arr = np.array(img_gray)
    img_gray_arr_flat = img_gray_arr.reshape(1, -1)

    # Normalize pixel values to be between 0 and 1
    img_gray_arr_flat_norm = img_gray_arr_flat / 255.0
    img_processed = img_gray_arr_flat_norm

    # Make prediction with SVM model
    prediction = svm.predict(img_processed)[0]

    # Get class name for prediction
    class_names = ["Very_Mild_Demented", "Mild_Demented",
                   "Moderate_Demented", "Non_Demented"]
    class_name = class_names[prediction]
    result = "Level Of : " + class_name

    return render_template('alzheimerdetection.html', image=img, prediction=prediction.tolist(), result=result)


@app.route('/send_mail', methods=['GET', 'POST'])
def send_mail():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # email body
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"

        # send email
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


@app.route('/stop', methods=['POST'])
def shutdown():
    # Check if the request is coming from a trusted source
    if request.remote_addr != '127.0.0.1':
        return "Unauthorized", 403

    # Shut down the server
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
    return render_template('blog3.html')


@app.route('/alzheimerdetection')
def alzheimerdetection():
    return render_template('alzheimerdetection.html')


if __name__ == '__main__':
    app.run(debug=True)
