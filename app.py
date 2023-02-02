from flask import Flask, request, make_response, send_file
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
    image_prompt = request.get_data(as_text=True)
    generated_images = do_inference(image_prompt, width, height, ckpt, samples)
    print(f'Images created for "{image_prompt}"')
    
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, mode='w') as zip:
        for i, image in enumerate(generated_images):
            image_file = io.BytesIO()
            image.save(image_file, 'PNG')
            image_file.seek(0)
            zip.writestr(f'image-{i}.png', image_file.read())
        details = {'prompt': image_prompt, 'ckpt': ckpt}
        details_file = io.StringIO()
        for key, value in details.items():
            details_file.write(f'{key}: {value}\n')
        details_file.seek(0)
        zip.writestr('details.txt', details_file.getvalue().encode('utf-8'))
    
    in_memory_zip.seek(0)
    response = make_response(in_memory_zip.read())
    response.headers["Content-Disposition"] = f"attachment; filename={image_prompt} images.zip"
    response.mimetype = 'application/zip'

    return response

if __name__ == '__main__':
    app.run(host='192.168.86.28', port=5000)