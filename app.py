from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import nltk
import pandas as pd
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
