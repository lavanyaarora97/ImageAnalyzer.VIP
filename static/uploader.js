    let selectedFiles = [];

    function handleFiles(files) {
        selectedFiles = selectedFiles.concat(Array.from(files)); // Combine old and new files
        const fileList = document.getElementById("file-list");
        fileList.innerHTML = ''; // Clear the list to redraw it
        selectedFiles.forEach(file => {
            const li = document.createElement("li");
            li.textContent = file.name;
            li.style.padding = '5px 0';
            fileList.appendChild(li);
        });
        document.getElementById("classify-btn").style.display = 'block'; // Make sure this ID matches your submit button for classification
        document.getElementById("detect-btn").style.display = 'block';  // Make sure this ID matches your submit button for detection
    }

    document.getElementById('drop-area').addEventListener('dragover', (event) => {
        event.stopPropagation();
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
        event.target.style.backgroundColor = '#f0f0f0'; // Highlight color on drag over
    });

    document.getElementById('drop-area').addEventListener('dragleave', (event) => {
        event.target.style.backgroundColor = 'transparent'; // Revert color on drag leave
    });

    document.getElementById('drop-area').addEventListener('drop', (event) => {
        event.stopPropagation();
        event.preventDefault();
        const files = event.dataTransfer.files;
        handleFiles(files);
        event.target.style.backgroundColor = 'transparent'; // Revert color after dropping
    });

    document.querySelector('form').onsubmit = function(e) {
        e.preventDefault();
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('file', file);
        });

        fetch(window.location.pathname, {  // Use the current path for the POST request
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            const results = document.getElementById('detection-results');
            results.innerHTML = ''; // Clear previous results
            data.forEach(detections => {
                const detectionDiv = document.createElement('div');
                detectionDiv.textContent = 'Detected Objects:';
                detections.forEach(det => {
                    const detItem = document.createElement('p');
                    detItem.textContent = `Object Class: ${det.class_id}, Confidence: ${det.score}, Coordinates: (${det.x1}, ${det.y1}, ${det.x2}, ${det.y2})`;
                    detectionDiv.appendChild(detItem);
                });
                results.appendChild(detectionDiv);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing your request. Please try again.');
        });
    };