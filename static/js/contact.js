// Select form and form elements
const form = document.querySelector('#contact-form');
const nameInput = document.querySelector('#name');
const emailInput = document.querySelector('#email');
const messageInput = document.querySelector('#message');
const submitButton = document.querySelector('#submit-button');

// Regular expressions for email validation
const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// Add event listener to form submission
form.addEventListener('submit', function(event) {
  event.preventDefault(); // Prevent default form submission

  // Validate form fields
  let errors = [];
  if (nameInput.value.trim() === '') {
    errors.push('Name is required.');
  }
  if (emailInput.value.trim() === '') {
    errors.push('Email is required.');
  } else if (!emailRegex.test(emailInput.value.trim())) {
    errors.push('Please enter a valid email address.');
  }
  if (messageInput.value.trim() === '') {
    errors.push('Message is required.');
  }

  // If there are errors, display them to the user
  if (errors.length > 0) {
    const errorDiv = document.querySelector('#error-message');
    errorDiv.innerHTML = '';
    errors.forEach(function(error) {
      const errorNode = document.createElement('div');
      errorNode.classList.add('error');
      errorNode.innerHTML = error;
      errorDiv.appendChild(errorNode);
    });
    return;
  }

  // If there are no errors, submit the form via AJAX
  const xhr = new XMLHttpRequest();
  xhr.open('POST', 'process-contact.php', true);
  xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
  xhr.onload = function() {
    if (this.status === 200) {
      // Display success message to the user
      const successDiv = document.querySelector('#success-message');
      successDiv.innerHTML = '<div class="success">Thank you for your message!</div>';

      // Clear form fields
      nameInput.value = '';
      emailInput.value = '';
      messageInput.value = '';
    } else {
      // Display error message to the user
      const errorDiv = document.querySelector('#error-message');
      errorDiv.innerHTML = '<div class="error">There was an error submitting your message. Please try again.</div>';
    }
  };
  xhr.send('name=' + encodeURIComponent(nameInput.value.trim()) + '&email=' + encodeURIComponent(emailInput.value.trim()) + '&message=' + encodeURIComponent(messageInput.value.trim()));
});
