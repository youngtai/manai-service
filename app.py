from flask import Flask, request
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

@app.route("/inference")
def inference():
    return "Inference complete!"

if __name__ == '__main__':
    app.run()