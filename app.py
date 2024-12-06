from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import nltk
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import open_clip
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import os


print("LOADING PICKLED DATA (wait one minute)")
df = pd.read_pickle('image_embeddings.pickle')
normalized_embeddings = np.stack(df['embedding'].to_numpy())
print("LOADED ! Now creating model.")
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()
print("DONE! Now loading images for PCA")
def load_images(image_dir="coco_images_resized", max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

def nearest_neighbors(query_embedding, embeddings, top_k=5):
    distances = euclidean_distances(query_embedding.reshape(1, -1), embeddings).flatten()
    nearest_indices = np.argsort(distances)[:top_k]
    
    return nearest_indices, distances

train_images, train_image_names = load_images(max_images=2000, target_size=(224, 224))
flat_images = np.array([img.flatten() for img in train_images])
print(f"Loaded {len(train_images)} images for PCA training.")
transform_images, transform_image_names = load_images(max_images=10000, target_size=(224, 224))
transform_images_flat = np.array([img.flatten() for img in transform_images[:10000]])
print(f"Loaded {len(transform_images)} images for transformation.")

print("Done! Now opening flask application.")
app = Flask(__name__)


@app.route('/images/<path:filename>')
def serve_image(filename):
    image_dir = os.path.join(app.root_path, 'coco_images_resized') 
    return send_from_directory(image_dir, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    queryrec = request.form.get('query')
    hybrid_weight = request.form.get('hybridWeight')
    query_type = request.form.get('queryType')
    image_file = request.files.get('imageQuery') 
    pcaK = request.form.get('pcaK')
    embed_type = request.form.get('embedType')
    if query_type == "text-query":
        text = tokenizer([queryrec])
        query = F.normalize(model.encode_text(text)).detach().numpy()
    elif query_type == "image-query":
        if image_file is None:
            return jsonify({'error': 'Image file needed for image query.'}), 400
        image = preprocess(Image.open(image_file.stream)).unsqueeze(0)
        query = F.normalize(model.encode_image(image)).detach().numpy()
    else:
        print("hyb")
        if image_file is None:
            return jsonify({'error': 'Image file needed for hybrid query.'}), 400
        image = preprocess(Image.open(image_file.stream)).unsqueeze(0)
        image_query = F.normalize(model.encode_image(image))
        text = tokenizer([queryrec])
        text_query = F.normalize(model.encode_text(text))
        lam  = float(hybrid_weight)
        query = F.normalize(lam * text_query + (1.0 - lam) * image_query).detach().numpy()

    print(pcaK, embed_type)
    if query_type == "image-query" and embed_type == "pca":
        print("pca!")
        k = int(pcaK)
        pca = PCA(n_components=k)
        pca.fit(transform_images_flat)
        reduced_embeddings = pca.transform(transform_images_flat)
        
        img = Image.open(image_file.stream)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((224, 224))  # Resize
        qimage = (np.asarray(img, dtype=np.float32) / 255.0).flatten()
        #not actually cosine similarities but to keep the same variable name when using pca or clip
        top_indices, cosine_similarities = nearest_neighbors(pca.transform(qimage.reshape(1,-1)), reduced_embeddings)

        top_images = [
        {
            'path': f"/images/{transform_image_names[index]}",
            'similarity': float(cosine_similarities[index])
        }
        for index in top_indices
        ]
        
    else:
        cosine_similarities = np.dot(normalized_embeddings, query.T).squeeze()
        top_indices = np.argsort(cosine_similarities)[-5:][::-1] 

        top_images = [
        {
            'path': f"/images/{df.iloc[index]['file_name']}",
            'similarity': float(cosine_similarities[index])
        }
        for index in top_indices
        ]

    return jsonify({'message': 'Success', 'query': queryrec, 'hybridWeight': hybrid_weight, 'queryType': query_type, 'topImages': top_images})


if __name__ == '__main__':
    app.run(port=3000, debug=True)
