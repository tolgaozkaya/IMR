import unittest
from flask import Flask, request
from PIL import Image
import numpy as np
from app import detect
from io import BytesIO
from werkzeug.datastructures import FileStorage

class TestDetect(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True

    def test_detect(self):
        with self.app.test_request_context('/detect', method='POST'):
            # Create a sample image
            img = Image.new('RGB', size=(256, 256))
            img_array = np.array(img)
            img_array = img_array.reshape((1, 256, 256, 3))
            img_array = img_array / 255.0
            
            # Set up a mock file object with the image data
            file = FileStorage(stream=img_array.tobytes(), filename='image.jpg', content_type='image/jpeg')
            request.files['image'] = file
            
            # Call the detect function and check the response
            with self.app.test_client() as client:
                response = client.post('/detect', data={'image': file})
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'Brain tumor not detected', response.data)

if __name__ == '__main__':
    unittest.main()
