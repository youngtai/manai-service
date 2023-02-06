from flask import Flask, request, make_response
import datetime
import json
from optimizedSD.text2image import do_inference
from werkzeug.utils import secure_filename
import zipfile
import io
from flask_cors import CORS

ALLOWED_EXTENSIONS = {'zip'}

app = Flask(__name__)
CORS(app)

checkpoints = {
    'base',
    'epoch=000071.ckpt',
    'epoch=000142.ckpt',
    'epoch=000214.ckpt',
    'epoch=000285.ckpt',
    'epoch=000357.ckpt'
}

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
    ckpt = request.args.get('ckpt', 'base')
    width = request.args.get('w', '512')
    height = request.args.get('h', '512')
    samples = request.args.get('samples', 4)
    sampler = request.args.get('sampler', 'plms')
    seed = request.args.get('seed', None)
    image_prompt = request.get_data(as_text=True)
    generated_images, images_details = do_inference(image_prompt, width, height, ckpt, samples, sampler, seed)
    print(f'Images created for "{image_prompt}"')
    
    in_memory_zip = get_in_memory_zip(generated_images, images_details)

    response = make_response(in_memory_zip)
    response.headers["Content-Disposition"] = f"attachment; filename=generated-images.zip"
    response.mimetype = 'application/zip'

    return response

def get_in_memory_zip(generated_images, images_details):
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, mode='w') as zip:
        for i, image in enumerate(generated_images):
            image_file = get_image_file(image)
            zip.writestr(f'image-{i}.png', image_file.read())

            details_file = get_details_file(images_details, i)
            zip.writestr(f'image-{i}.json', details_file)
    in_memory_zip.seek(0)
    return in_memory_zip.read()

def get_image_file(image):
    image_file = io.BytesIO()
    image.save(image_file, 'PNG')
    image_file.seek(0)
    return image_file

def get_details_file(images_details, i):
    details = images_details[i]
    details_file = json.dumps(details)
    return details_file

if __name__ == '__main__':
    app.run(host='192.168.86.28', port=5000)