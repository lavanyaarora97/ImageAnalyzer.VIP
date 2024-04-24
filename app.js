document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(event) {
        event.preventDefault();  // Prevent the default form submission behavior
        const formData = new FormData(this);
        fetch(form.action, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())  // Assuming the server responds with JSON
        .then(data => {
            displayResults(data);  // Function to handle the display of results
        })
        .catch(error => console.error('Error:', error));
    });
});

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';  // Clear previous results
    if (data.error) {
        resultsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
    } else {
        // Example: Displaying a result message and image
        resultsDiv.innerHTML = `<p>Detected: ${data.detectedLabel}</p><img src="${data.imageUrl}" alt="Processed Image"/>`;
    }
}

function showLoadingIndicator() {
    const loadingDiv = document.getElementById('loading');
    loadingDiv.style.display = 'block';  // Make the loading div visible
}

function hideLoadingIndicator() {
    const loadingDiv = document.getElementById('loading');
    loadingDiv.style.display = 'none';  // Hide the loading div
}

// Add these inside your fetch().then().catch() chain
.catch(error => {
    console.error('Error uploading file:', error);
    alert('Failed to upload the file. Please try again.');
});

function addZoomFeature() {
    const image = document.querySelector('img');
    image.addEventListener('click', function(event) {
        // Simple zoom feature toggle
        if (this.style.transform === 'scale(1.5)') {
            this.style.transform = 'scale(1)';
        } else {
            this.style.transform = 'scale(1.5)';
        }
    });
}

