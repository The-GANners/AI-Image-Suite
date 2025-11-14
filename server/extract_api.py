from flask import Blueprint, request, jsonify
import os
import tempfile
import uuid
from pathlib import Path
import subprocess

# Get the correct Python executable for the current environment
PYTHON_EXEC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'myenv', 'Scripts', 'python.exe'))

extract_bp = Blueprint('extract_bp', __name__)

@extract_bp.route('/api/extract-watermark', methods=['POST'])
def extract_watermark_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    text = request.form.get('text', None)

    # Save uploaded image to temp file
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, f"uploaded_{uuid.uuid4().hex}.png")
    image_file.save(img_path)

    # Build command
    script_path = os.path.join(os.path.dirname(__file__), 'wm_extract.py')
    # Use the environment's Python executable
    cmd = [PYTHON_EXEC, script_path, img_path]
    if text:
        cmd += ['--text', text]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        error = result.stderr
        if result.returncode != 0:
            return jsonify({'error': error or output or 'Extraction failed.'}), 500
        return jsonify({'output': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(img_path)
            os.rmdir(temp_dir)
        except Exception:
            pass
