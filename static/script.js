document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    let query = document.getElementById('query').value;
    let imageQuery = document.getElementById('image-query').files[0]; 
    let hybridWeight = document.getElementById('hybrid-weight').value;
    let queryType = document.getElementById('query-type').value;

    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    let formData = new FormData();
    formData.append('query', query);
    formData.append('hybridWeight', hybridWeight);
    formData.append('queryType', queryType);
    
    if (imageQuery) {
        formData.append('imageQuery', imageQuery);
    }

    fetch('/search', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = ''; 
    
        const flexContainer = document.createElement('div');
        flexContainer.className = 'flex flex-col items-center gap-8';
    
        data.topImages.forEach(image => {
            const imgElement = document.createElement('img');
            imgElement.src = image.path;
            imgElement.alt = `Similarity: ${image.similarity.toFixed(2)}`;
            imgElement.className = 'w-128 h-auto rounded shadow-lg';
    
            const caption = document.createElement('p');
            caption.textContent = `Similarity: ${image.similarity.toFixed(2)}`;
            caption.className = 'text-sm text-center';
    
            const container = document.createElement('div');
            container.className = 'flex flex-col items-center'; 
            container.appendChild(imgElement);
            container.appendChild(caption);
    
            flexContainer.appendChild(container);
        });
    
        resultsDiv.appendChild(flexContainer);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});