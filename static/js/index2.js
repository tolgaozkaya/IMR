const uploadButton = document.querySelector('.custom-file-upload');
const fileInput = document.querySelector('input[type="file"]');
const detectButton = document.querySelector('button[type="submit"]');
const resultImage = document.querySelector('.result-container img');
const inputImage = document.querySelector('.input-container img');

uploadButton.addEventListener('click', () => {
  fileInput.click();
});

detectButton.addEventListener('click', async (event) => {
  event.preventDefault();

  const formData = new FormData();
  formData.append('image', fileInput.files[0]);

  const response = await fetch('/predict', {
    method: 'POST',
    body: formData
  });

  if (response.ok) {
    const data = await response.json();
    resultImage.src = data.result;
  } else {
    console.error('An error occurred while detecting the brain tumor.');
  }
});
