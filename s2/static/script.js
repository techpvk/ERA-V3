document.querySelectorAll('input[name="animal"]').forEach((radio) => {
    radio.addEventListener('change', function() {
        const animal = this.value;
        const img = document.getElementById('animal-image');
        img.src = `/static/images/${animal}.jpg`;
        img.style.display = 'block';
    });
});

function uploadFile() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const fileInfoDiv = document.getElementById('file-info');
            fileInfoDiv.innerHTML = `Name: ${data.name}<br>Size: ${data.size} bytes<br>Type: ${data.type}`;
        })
        .catch(error => console.error('Error:', error));
    }
}
