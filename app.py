from flask import Flask, request
from optimizedSD.text2image import do_inference
from werkzeug.utils import secure_filename
import zipfile
from flask_cors import CORS

ALLOWED_EXTENSIONS = {'zip'}

app = Flask(__name__)
CORS(app)

@app.route("/file-upload", methods=['POST'])
def file_upload():
    if len(request.files) == 0 or 'images' not in request.files:
        print('No file called "images" in request')
        return 'No file called "images" in request'
    file = request.files.get('images')
    file.save(f'/home/youngtai/dev/{secure_filename(file.filename)}')
    return "Received images!"

@app.route("/training")
def training():
    return "Training started!"

@app.route("/inference", methods=['POST'])
def inference():
    image_prompt = request.get_data(as_text=True)
    generated_images = do_inference(image_prompt)
    print(f'Images created for "{image_prompt}"')
    return generated_images

if __name__ == '__main__':
    app.run()