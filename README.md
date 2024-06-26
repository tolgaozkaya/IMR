# IMR
This is a Flask web application that provides several functionalities, including brain tumor detection and Alzheimer's disease prediction. It also allows users to send emails and provides information about the project's services and blog posts.

![IMR](https://res.cloudinary.com/da9md6gyy/image/upload/v1685396122/screenshot_m1nimy.png)

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3
- Flask
- TensorFlow
- Scikit-learn
- PIL (Python Imaging Library)
- NumPy
- Joblib
- TensorFlow Hub

You can install the dependencies by running the following command:

- pip install -r requirements.txt

## Usage

**1. Clone the repository:**

- git clone <https://github.com/tolgaozkaya/IMR.git>
- cd IMR

**2. Set up the virtual environment (optional but recommended):**

- python -m venv venv
- source venv/bin/activate # For Linux/macOS
- venv\Scripts\activate # For Windows

**3. Install the required dependencies:**

- pip install -r requirements.txt

**4. Run the Flask application:**

- python app.py

**5. Access the application in your browser at `http://localhost:5000`.**

## Functionality

The Flask web application provides the following functionalities:

### Brain Tumor Detection

Upload an image of a brain scan to detect the presence of a brain tumor.

### Alzheimer's Disease Prediction

Upload an image of a brain scan to predict the level of Alzheimer's disease.

### Send Email

Fill out a contact form to send an email to the project team.

### Information Pages

Access various information pages about the project, services, and blog posts.

## File Structure

The file structure of the project is as follows:

- `app.py`: The main Flask application file containing route handlers and configuration.

- `templates/`: Directory containing HTML templates for rendering web pages.

- `static/`: Directory containing static files such as CSS stylesheets, JavaScript files, and images.

- `svm_model.joblib`: Pretrained SVM model for Alzheimer's disease prediction.

- `best_model.pt`: Pretrained YOLOV8 model for brain tumor detection.

- `EfficientNetB7.h5`: Pretrained Keras model for brain tumor detection.

- `README.md`: This file providing information about the project.

## Contributing

Contributions to the project are welcome! If you find any issues or want to add new features, feel free to open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

