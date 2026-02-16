from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
from werkzeug.utils import secure_filename
from DATA import overlay_mask_on_image

UPLOAD_FOLDER = os.path.join('static', 'data', 'uploads')
ALLOWED_EXT = {'.png', '.jpg', '.jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No image field in request', 400
    image = request.files['image']
    mask_file = request.files.get('mask')

    if image.filename == '' or not allowed_file(image.filename):
        return 'Invalid or missing image file', 400

    image_name = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    mask_path = None
    if mask_file and mask_file.filename != '' and allowed_file(mask_file.filename):
        mask_name = secure_filename(mask_file.filename)
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_name)
        mask_file.save(mask_path)

    try:
        out_path = overlay_mask_on_image(image_path=image_path, mask_path=mask_path)
    except Exception as e:
        return f'Processing error: {e}', 500

    # return the annotated file for download/view
    rel = os.path.relpath(out_path)
    return redirect(url_for('uploaded_file', filename=os.path.basename(rel)))


@app.route('/static/trained/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join('static', 'trained'), filename)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
