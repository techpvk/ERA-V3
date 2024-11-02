function selectInputType(type) {
    const uploadSection = document.getElementById('uploadSection');
    const inputForm = document.getElementById('inputForm');
    
    // Show the form
    inputForm.classList.remove('hidden');
    
    // Clear previous content
    uploadSection.innerHTML = '';
    
    // Create appropriate input based on type
    switch(type) {
        case 'text':
            uploadSection.innerHTML = `
                <textarea 
                    name="content" 
                    placeholder="Enter your text here..."
                    rows="6"
                    style="width: 100%; padding: 1rem; margin-bottom: 1rem;"
                ></textarea>
            `;
            break;
            
        case 'image':
            uploadSection.innerHTML = `
                <input 
                    type="file" 
                    name="content"
                    accept="image/*"
                    style="margin-bottom: 1rem;"
                >
                <div id="imagePreview"></div>
            `;
            break;
            
        case 'audio':
            uploadSection.innerHTML = `
                <input 
                    type="file" 
                    name="content"
                    accept="audio/*"
                    style="margin-bottom: 1rem;"
                >
                <audio id="audioPreview" controls style="display: none;"></audio>
            `;
            break;
            
        case '3d':
            uploadSection.innerHTML = `
                <input 
                    type="file" 
                    name="content"
                    accept=".obj,.glb,.gltf"
                    style="margin-bottom: 1rem;"
                >
                <div id="modelPreview"></div>
            `;
            break;
    }
    
    // Add event listeners for previews
    setupPreviewListeners(type);
}

function setupPreviewListeners(type) {
    const input = document.querySelector('input[type="file"]');
    if (!input) return;
    
    input.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        switch(type) {
            case 'image':
                const imagePreview = document.getElementById('imagePreview');
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.innerHTML = `
                        <img src="${e.target.result}" 
                             style="max-width: 100%; margin-top: 1rem;">
                    `;
                };
                reader.readAsDataURL(file);
                break;
                
            case 'audio':
                const audioPreview = document.getElementById('audioPreview');
                audioPreview.src = URL.createObjectURL(file);
                audioPreview.style.display = 'block';
                break;
        }
    });
}

// Handle form submission
document.getElementById('contentForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            window.location.href = `/results/${result.id}`;
        } else {
            throw new Error('Processing failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your content');
    }
}); 