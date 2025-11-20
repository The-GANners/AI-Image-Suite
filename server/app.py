import base64
import io
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional
import sys
import time
import numpy as np
import cv2
from PIL import Image as PILImage
import re  # NEW
from PIL import Image
import logging  # NEW

# NEW: force all prints to ASCII-safe to prevent Windows console Unicode errors
import builtins as _builtins
_original_print = print
def _ascii_print(*args, **kwargs):
    try:
        msg = " ".join(str(a) for a in args)
        msg = msg.encode('ascii', 'replace').decode('ascii')
        _original_print(msg, **kwargs)
    except Exception:
        try:
            _original_print(*args, **kwargs)
        except Exception:
            pass
_builtins.print = _ascii_print

# NEW: ASCII-safe logger setup (moved up so it's available for early logs)
class AsciiSafeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Sanitize the message to ASCII but keep format placeholders intact
            if isinstance(record.msg, (bytes, bytearray)):
                record.msg = record.msg.decode('ascii', 'replace')
            else:
                record.msg = str(record.msg).encode('ascii', 'replace').decode('ascii')

            # Preserve numeric types to satisfy %d / %.4f, only sanitize non-numeric args
            if record.args:
                new_args = []
                for a in record.args:
                    if isinstance(a, (int, float)):
                        new_args.append(a)
                    elif isinstance(a, (bytes, bytearray)):
                        new_args.append(a.decode('ascii', 'replace'))
                    else:
                        new_args.append(str(a).encode('ascii', 'replace').decode('ascii'))
                record.args = tuple(new_args)
        except Exception:
            # Do not modify args on failure to avoid formatter errors
            pass
        return True

logger = logging.getLogger("ai_image_suite")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    _h.addFilter(AsciiSafeFilter())
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Silence logging formatter exceptions in production-like runs
logging.raiseExceptions = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Fix for compatibility with newer Werkzeug versions
import werkzeug
if not hasattr(werkzeug.urls, 'url_quote'):
    werkzeug.urls.url_quote = werkzeug.urls.quote

# Add torch import
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add a guard to avoid duplicate logs when reloader is active
RUN_MAIN = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

# Add Module-2 to Python path
MODULE2_PATH = Path(__file__).parent.parent.parent / "Module-2"
if str(MODULE2_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE2_PATH))

# FIXED: Import config and other server modules BEFORE adding watermarker path
# This prevents importing the wrong config.py from Watermark-UI
from config import (
    DF_GAN_PATH, DF_GAN_CODE_PATH, DF_GAN_SRC_PATH,
    CUB_WEIGHTS, COCO_WEIGHTS, get_data_dir, validate_paths, get_config_info
)
from dfgan_wrapper import DFGANGenerator

# Import the vehicle generator
from vehicle_gan import is_vehicle_prompt, generate_vehicle_image

# NEW: Import the plant generator
from plant_gan import is_plant_prompt, generate_plant_image

# Import other modules
from stylegan_wrapper import StyleGANGenerator
from blend_utils import blend_images

# NEW: Add Watermark-UI to Python path AFTER importing server config
WATERMARK_UI_PATH = Path(__file__).parent.parent.parent / "Watermark-UI"
if str(WATERMARK_UI_PATH) not in sys.path:
    sys.path.insert(0, str(WATERMARK_UI_PATH))

# FIXED: Import WaterMarker after adding path
try:
    watermarker_path = WATERMARK_UI_PATH / "FreeMark" / "tools"
    if str(watermarker_path) not in sys.path:
        sys.path.insert(0, str(watermarker_path))
    from watermarker import WaterMarker
    WATERMARK_AVAILABLE = True
    if RUN_MAIN:
        logger.info("[SERVER] WaterMarker imported from: %s", WATERMARK_UI_PATH)
except Exception as e:
    WATERMARK_AVAILABLE = False
    if RUN_MAIN:
        logger.error("[SERVER] Failed to import WaterMarker: %s", e)

if RUN_MAIN:
    logger.info("[SERVER] Added Module-2 path: %s", MODULE2_PATH)

# FIXED: Import evaluator after Module-2 is in path
try:
    if str(MODULE2_PATH) not in sys.path:
        sys.path.insert(0, str(MODULE2_PATH))
    from image_prompt_evaluator import ImagePromptEvaluator
    EVALUATOR_AVAILABLE = True
    if RUN_MAIN:
        logger.info("[SERVER] ImagePromptEvaluator imported successfully")
except ImportError as e:
    EVALUATOR_AVAILABLE = False
    if RUN_MAIN:
        logger.error("[SERVER] Failed to import ImagePromptEvaluator: %s", e)

# Add auth_routes and gallery_routes imports here
# Import auth routes - use importlib to avoid conflict with DF-GAN models
import importlib.util
from pathlib import Path

server_dir = Path(__file__).parent

# Load models.py explicitly
models_path = server_dir / 'models.py'
spec = importlib.util.spec_from_file_location("server_models", models_path)
server_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_models)
init_db = server_models.init_db
User = server_models.User
GeneratedImage = server_models.GeneratedImage
EvaluatedImage = server_models.EvaluatedImage
WatermarkedImage = server_models.WatermarkedImage
Session = server_models.Session

# Load auth_routes.py explicitly
auth_routes_path = server_dir / 'auth_routes.py'
spec_auth = importlib.util.spec_from_file_location("server_auth_routes", auth_routes_path)
auth_routes_module = importlib.util.module_from_spec(spec_auth)
spec_auth.loader.exec_module(auth_routes_module)
auth_bp = auth_routes_module.auth_bp

# Load gallery_routes.py explicitly
gallery_routes_path = server_dir / 'gallery_routes.py'
spec_gallery = importlib.util.spec_from_file_location("server_gallery_routes", gallery_routes_path)
gallery_routes_module = importlib.util.module_from_spec(spec_gallery)
spec_gallery.loader.exec_module(gallery_routes_module)
gallery_bp = gallery_routes_module.gallery_bp

# Load extract_api blueprint
extract_api_path = server_dir / 'extract_api.py'
spec_extract = importlib.util.spec_from_file_location("server_extract_api", extract_api_path)
extract_api_module = importlib.util.module_from_spec(spec_extract)
spec_extract.loader.exec_module(extract_api_module)
extract_bp = extract_api_module.extract_bp

app = Flask(__name__)
CORS(app, 
     origins=["http://localhost:3000"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])  # Explicit CORS for React app

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(gallery_bp)
app.register_blueprint(extract_bp)

# Initialize database on first import
try:
    init_db()
    if RUN_MAIN:
        logger.info("[SERVER] Database initialized successfully")
except Exception as e:
    if RUN_MAIN:
        logger.warning("[SERVER] Database initialization warning: %s", e)

# Helper function to get user from request
def get_user_from_request():
    """Extract user from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    user_id = User.verify_token(token)
    
    if not user_id:
        return None
    
    session = Session()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        return user
    finally:
        session.close()

# Helper function to save generated image to database
def save_generated_image(user_id, prompt, img_path, dataset):
    """Save generated image metadata to database"""
    try:
        # Create directory for permanent storage
        generated_dir = Path(__file__).parent.parent / 'generated_images'
        generated_dir.mkdir(exist_ok=True)
        
        # Generate permanent filename
        permanent_filename = f"{uuid.uuid4().hex}.png"
        permanent_path = generated_dir / permanent_filename
        
        # Copy image to permanent location
        shutil.copy(img_path, permanent_path)
        
        # Save to database
        session = Session()
        try:
            generated_img = GeneratedImage(
                user_id=user_id,
                prompt=prompt,
                file_path=str(permanent_path),
                dataset=dataset
            )
            session.add(generated_img)
            session.commit()
            return permanent_filename
        finally:
            session.close()
    except Exception as e:
        logger.error("Error saving generated image: %s", e)
        return None

# Helper function to save watermarked image to database
def save_watermarked_image(user_id, original_path, watermarked_pil, watermark_text, position, opacity):
    """Save watermarked image and metadata to database"""
    try:
        # Create directory for permanent storage
        watermarked_dir = Path(__file__).parent.parent / 'watermarked_images'
        watermarked_dir.mkdir(exist_ok=True)
        
        # Generate permanent filename
        permanent_filename = f"{uuid.uuid4().hex}.png"
        permanent_path = watermarked_dir / permanent_filename
        
        # Save watermarked PIL image
        watermarked_pil.save(permanent_path, 'PNG')
        
        # Save to database
        session = Session()
        try:
            watermarked_img = WatermarkedImage(
                user_id=user_id,
                original_image_path=original_path,
                watermarked_image_path=str(permanent_path),
                watermark_text=watermark_text,
                watermark_position=position,
                watermark_opacity=int(opacity * 100)  # Convert to percentage
            )
            session.add(watermarked_img)
            session.commit()
            return permanent_filename
        finally:
            session.close()
    except Exception as e:
        logger.error("Error saving watermarked image: %s", e)
        return None

# Global variables
LAST_GEN = {"ts": 0, "prompt": "", "images": []}
stylegan_gen = StyleGANGenerator()
_evaluator = None

# Weak entities for blending
weak_entities = {
    "man": "human",
    "woman": "human",
    "boy": "human",
    "girl": "human",
    "child": "human",
    "children": "human",
    "dog": "dog",
    "cat": "cat",
    "tiger": "wild",
    "bear": "wild",
    "zebra": "wild",
    "giraffe": "wild",
}

def get_evaluator():
    global _evaluator
    if EVALUATOR_AVAILABLE and _evaluator is None:
        try:
            _evaluator = ImagePromptEvaluator()
            logger.info("[SERVER] ImagePromptEvaluator initialized")
        except Exception as e:
            logger.error("Failed to initialize evaluator: %s", e)
            return None
    return _evaluator

def detect_entity(prompt: str):
    for k, v in weak_entities.items():
        if k in prompt.lower():
            return v
    return None

def is_cub_prompt(prompt: str) -> bool:
    p = (prompt or '').lower()
    keys = ['bird', 'sparrow', 'eagle', 'jay', 'owl', 'finch', 'feather', 'beak', 'wing', 'robin', 'parrot']
    return any(k in p for k in keys)

def ensure_paths_ok():
    """Validate that all required paths exist using the config module."""
    success, issues = validate_paths()
    if not success:
        raise FileNotFoundError(f"Path validation failed: {'; '.join(issues)}")
    
    return DF_GAN_SRC_PATH / 'sample.py'

def dfgan_generate(prompt: str, model_key: str, out_dir: Path, seed: Optional[int], steps: int, guidance: float) -> Path:
    """Execute DF-GAN inference using our wrapper for sample.py"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set model-specific paths using config module
    if model_key == 'CUB':
        weights = CUB_WEIGHTS
        data_dir = str(get_data_dir('birds'))
    else:
        weights = COCO_WEIGHTS
        data_dir = str(get_data_dir('coco'))
    
    logger.info("Using model: %s", weights)
    logger.info("Using data directory: %s", data_dir)
    logger.info("Prompt: %s", prompt)
    
    try:
        # Create or get the generator for this model
        use_cuda = torch.cuda.is_available()
        seed_value = seed if seed is not None and seed >= 0 else 100

        # Create generator for this specific model
        generator = DFGANGenerator(
            model_path=str(weights),
            data_dir=data_dir,
            use_cuda=use_cuda,
            seed=seed_value,
            steps=steps,
            guidance=guidance
        )

        # Generate the image
        img_path = generator.generate_image(prompt, out_dir)
        logger.info("Image generated at: %s", img_path)

        return img_path

    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)

def encode_b64(path: Path) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

@app.route('/api/check', methods=['GET'])
def check_setup():
    """Endpoint to check if the setup is correct"""
    success, issues = validate_paths()
    paths = get_config_info()
    
    # Return status
    if not success:
        return jsonify({
            'status': 'error',
            'issues': issues,
            'paths': paths
        }), 400
    
    return jsonify({
        'status': 'ok',
        'paths': paths
    })

# NEW: simple animal keyword set and helper for logging label
_ANIMAL_KEYWORDS = {
    'lion','tiger','leopard','cheetah','jaguar','panther','wolf','fox','bear','zebra',
    'giraffe','elephant','rhinoceros','rhino','hippopotamus','hippo','crocodile','alligator',
    'deer','moose','bison','buffalo','horse','cow','goat','sheep','camel','dog','puppy',
    'cat','kitten','panda','koala','kangaroo','otter','raccoon'
}
def _is_animal_prompt(p: str) -> bool:
    s = (p or '').lower()
    cleaned = re.sub(r'[^a-z0-9\s]+', ' ', s)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    padded = f" {cleaned} "
    return any(f" {kw} " in padded for kw in _ANIMAL_KEYWORDS)

@app.route('/api/generate', methods=['POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS':
        # Handle CORS preflight request
        response = jsonify({
            'message': 'CORS preflight response'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return response

    data = request.get_json()
    prompt = data.get('prompt', 'A dog')
    dataset = (data.get('dataset') or '').strip()
    batch_size = int(data.get('batchSize') or 1)
    seeds = data.get('seeds')
    steps = int(data.get('steps') or 50)
    guidance = float(data.get('guidance') or 7.5)

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    # Get authenticated user (optional - anonymous generation still allowed)
    current_user = get_user_from_request()
    if current_user:
        logger.info("[GENERATE] Authenticated user: %s", current_user.email)
    else:
        logger.info("[GENERATE] Anonymous generation")

    try:
        script_path = ensure_paths_ok()
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # If dataset is 'bird', force CUB model, otherwise check prompt
    if dataset == 'bird':
        model_key = 'CUB'
    elif dataset == 'coco':
        model_key = 'COCO'
    else:
        model_key = 'CUB' if is_cub_prompt(prompt) else 'COCO'

    logger.info("Using %s model for prompt '%s'", model_key, prompt)
    logger.info("Using script: %s", script_path)

    tmp_root = Path(tempfile.gettempdir()) / f'dfgan_{uuid.uuid4().hex}'
    images_b64: List[str] = []
    generated_paths: List[Path] = []

    try:
        for i in range(batch_size):
            seed = None
            if seeds and isinstance(seeds, list) and i < len(seeds):
                seed = seeds[i]

            # NEW: Skip BigGAN entirely if dataset is 'bird' (CUB)
            if dataset == 'bird':
                # BIRD dataset: ONLY use DF-GAN, no BigGAN interference
                img_path = dfgan_generate(
                    prompt, model_key, tmp_root,
                    seed if seed is not None and int(seed) >= 0 else None,
                    steps, guidance
                )
                images_b64.append(encode_b64(img_path))
                generated_paths.append(img_path)
                logger.info("[CUB] Generated bird image using DF-GAN only: %s", img_path)
            
            # COCO dataset: Check for specialized generators (Plant/Vehicle/Animal)
            elif is_plant_prompt(prompt):
                logger.info("[PLANT] Using plant generator for: %s", prompt)
                output_filename = uuid.uuid4().hex
                img_path, plant_type = generate_plant_image(prompt, tmp_root, output_filename)
                
                if img_path and os.path.exists(str(img_path)):
                    images_b64.append(encode_b64(img_path))
                    generated_paths.append(img_path)
                    logger.info("[PLANT] Plant image generated: %s (type=%s)", img_path, plant_type)
                else:
                    logger.warning("[PLANT] Plant generation failed; falling back to DF-GAN")
                    img_path = dfgan_generate(
                        prompt, model_key, tmp_root,
                        seed if seed is not None and int(seed) >= 0 else None,
                        steps, guidance
                    )
                    images_b64.append(encode_b64(img_path))
                    generated_paths.append(img_path)
            
            # Vehicle/Animal specialized handling
            elif is_vehicle_prompt(prompt) or _is_animal_prompt(prompt):
                label = 'Animal' if _is_animal_prompt(prompt) else 'Vehicle'
                logger.info("[%s] Using specialized generator for: %s", label, prompt)
                output_filename = uuid.uuid4().hex
                img_path, vehicle_type = generate_vehicle_image(prompt, tmp_root, output_filename)
                
                if img_path and os.path.exists(str(img_path)):
                    images_b64.append(encode_b64(img_path))
                    generated_paths.append(img_path)
                    logger.info("%s image generated: %s (type=%s)", label, img_path, vehicle_type)
                else:
                    logger.warning("%s generation failed; falling back to DF-GAN", label)
            
            # Default: DF-GAN for everything else
            else:
                img_path = dfgan_generate(
                    prompt, model_key, tmp_root,
                    seed if seed is not None and int(seed) >= 0 else None,
                    steps, guidance
                )

                # Load DF-GAN output as numpy array
                dfgan_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

                # Check if prompt needs StyleGAN entity
                entity_type = detect_entity(prompt)
                if entity_type:
                    entity_img = stylegan_gen.generate(entity_type, seed=np.random.randint(10000))
                    blended_img = blend_images(dfgan_img, entity_img, position=(64, 64))
                    # Overwrite DF-GAN file with blended image
                    cv2.imwrite(str(img_path), cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR))
                    logger.info("[BLEND] Entity blend applied for type=%s", entity_type)

                images_b64.append(encode_b64(img_path))
                generated_paths.append(img_path)

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        shutil.rmtree(tmp_root, ignore_errors=True)
        return jsonify({'error': str(e)}), 500

    # Save images to database for authenticated users
    if current_user:
        try:
            for img_path in generated_paths:
                save_generated_image(current_user.id, prompt, img_path, dataset or model_key)
            logger.info("[GENERATE] Saved %d images for user %s", len(generated_paths), current_user.email)
        except Exception as e:
            logger.warning("[GENERATE] Failed saving images: %s", e)
    
    # Publish event, cleanup, response
    try:
        LAST_GEN["ts"] = int(time.time() * 1000)
        LAST_GEN["prompt"] = prompt
        LAST_GEN["images"] = images_b64[:]
    except Exception:
        pass

    shutil.rmtree(tmp_root, ignore_errors=True)
    return jsonify({'images': images_b64, 'model': model_key})

@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to avoid 404s"""
    return '', 204

@app.route('/api/gen-events', methods=['GET'])
def gen_events():
    """Returns the last generation event"""
    return jsonify({
        "ts": LAST_GEN.get("ts", 0),
        "prompt": LAST_GEN.get("prompt", ""),
        "images": LAST_GEN.get("images", []),
    })

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    logger.info("[TEST] /api/test called method=%s", request.method)
    if request.method == 'POST':
        data = request.get_json()
        logger.info("[TEST] POST payload keys=%s", list(data.keys()) if isinstance(data, dict) else 'N/A')
        return jsonify({'status': 'success', 'method': 'POST', 'data': data})
    else:
        return jsonify({'status': 'success', 'method': 'GET', 'message': 'API is working'})

@app.route('/api/evaluate-image', methods=['POST', 'OPTIONS'])
def evaluate_single_image():
    logger.info("[EVAL] /api/evaluate-image method=%s", request.method)
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        logger.info("[EVAL] Handling CORS preflight")
        response = jsonify({'message': 'CORS preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    try:
        logger.info("[EVAL] Processing POST request")
        
        # Check content type
        if not request.is_json:
            logger.error("[EVAL] Request is not JSON, content-type: %s", request.content_type)
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            logger.error("[EVAL] No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        logger.info("[EVAL] JSON data keys: %s", list(data.keys()))
        
        image_b64 = data.get('image', '')
        prompt = data.get('prompt', '').strip()
        threshold = float(data.get('threshold', 0.22))
        
        logger.info("[EVAL] Prompt: '%s'", prompt[:50])
        logger.info("[EVAL] Threshold: %s", threshold)
        logger.info("[EVAL] Image data length: %d", len(image_b64))
        
        if not image_b64 or not prompt:
            logger.error("[EVAL] Missing image or prompt")
            return jsonify({'error': 'Image and prompt required'}), 400

        # Get evaluator
        evaluator = get_evaluator()
        if not evaluator:
            logger.error("[EVAL] Evaluator not available")
            return jsonify({'error': 'Evaluator not available - check server logs'}), 500

        # Decode base64 image
        try:
            if ',' in image_b64:
                image_b64 = image_b64.split(',', 1)[1]
            
            image_bytes = base64.b64decode(image_b64)
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.info("[EVAL] Image decoded size=%s", img.size)
        except Exception as e:
            logger.error("[EVAL] Image decode error: %s", e)
            return jsonify({'error': f'Invalid image: {e}'}), 400

        # Save to temp file
        tmp_dir = Path(tempfile.gettempdir()) / f'eval_{uuid.uuid4().hex}'

        tmp_dir.mkdir(parents=True, exist_ok=True)
        img_path = tmp_dir / 'image.png'
        img.save(str(img_path))
        logger.info("[EVAL] Image saved temp=%s", img_path)

        # Run Module-2 evaluation
        logger.info("[EVAL] Running evaluation...")
        try:
            result = evaluator.evaluate_image(str(img_path), prompt, threshold)
            logger.info("[EVAL] Complete percentage=%s quality=%s",
                        result.get('percentage_match'), result.get('quality'))
            
            if 'keyword_analysis' in result:
                logger.info("[EVAL] Keywords: %d analyzed", len(result['keyword_analysis']))
        except Exception as e:
            logger.error("[EVAL] Evaluation failed: %s", e)
            import traceback
            traceback.print_exc()
            result = {'error': str(e)}

        # Cleanup
        try:
            img_path.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except:
            pass

        if 'error' in result:
            logger.error("[EVAL] Returning error: %s", result['error'])
            return jsonify(result), 500
            
        logger.info("[EVAL] Returning response")
        return jsonify(result)
        
    except Exception as e:
        logger.error("[EVAL] Critical error: %s", e)
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# NEW: normalize legacy/varied payloads into a single internal format
def _normalize_watermark_payload(payload: dict) -> dict:
    # Some UIs send { images, options: { ... } }
    opts = payload.get('options') or {}
    mode = (payload.get('mode') or opts.get('mode') or 'image').strip().lower()
    # Text mode aliases
    text = payload.get('text') or opts.get('text') or payload.get('watermarkText') or opts.get('watermarkText')
    text_size = payload.get('textSize') or opts.get('textSize') or payload.get('fontSize') or opts.get('fontSize') or 32
    text_color = payload.get('textColor') or opts.get('textColor') or payload.get('color') or opts.get('color') or '#FFFFFF'

    # Position aliases (support center synonyms)
    pos = (payload.get('pos') or opts.get('pos') or payload.get('position') or opts.get('position') or 'SE')
    pos = (str(pos) or 'SE').strip().upper()
    if pos in ('CENTER', 'CENTRE', 'MIDDLE'):
        pos = 'C'

    # Padding: prefer nested payload.padding if provided; fallback to aliases
    p_obj = payload.get('padding') or opts.get('padding') or {}
    if isinstance(p_obj, dict) and ('x' in p_obj or 'y' in p_obj):
        try:
            padx = int(p_obj.get('x', 20))
        except Exception:
            padx = 20
        try:
            pady = int(p_obj.get('y', 5))
        except Exception:
            pady = 5
        unit_x = p_obj.get('xUnit') or p_obj.get('unitX') or 'px'
        unit_y = p_obj.get('yUnit') or p_obj.get('unitY') or 'px'
    else:
        padx = payload.get('padx') or opts.get('padx') or (payload.get('padX') or opts.get('padX') or payload.get('xPad') or opts.get('xPad') or 20)
        pady = payload.get('pady') or opts.get('pady') or (payload.get('padY') or opts.get('padY') or payload.get('yPad') or opts.get('yPad') or 5)
        unit_x = payload.get('xUnit') or opts.get('xUnit') or payload.get('unitX') or opts.get('unitX') or 'px'
        unit_y = payload.get('yUnit') or opts.get('yUnit') or payload.get('unitY') or opts.get('unitY') or 'px'
    # Sanitize units to 'px' or '%'
    unit_x = '%' if str(unit_x).strip().lower() in ('%', 'percent', 'percentage') else 'px'
    unit_y = '%' if str(unit_y).strip().lower() in ('%', 'percent', 'percentage') else 'px'

    # Scale/opacity/rotation aliases
    scale = payload.get('scale')
    if scale is None:
        scale = opts.get('scale')
    if scale is None:
        scale = payload.get('autoResizeWatermark', opts.get('autoResizeWatermark', True))

    raw_opacity = payload.get('opacity')
    if raw_opacity is None:
        raw_opacity = opts.get('opacity')
    if raw_opacity is None:
        raw_opacity = payload.get('opacityPercent', opts.get('opacityPercent', 50))
    try:
        opacity = float(raw_opacity)
        # Accept 0..1 or 0..100
        if opacity > 1.0:
            opacity = opacity / 100.0
    except Exception:
        opacity = 0.5

    # Rotation can be provided as degrees; accept string or number
    raw_rotation = payload.get('rotation')
    if raw_rotation is None:
        raw_rotation = opts.get('rotation')
    try:
        rotation = int(float(raw_rotation)) if raw_rotation is not None else 0
        # Clamp to -180..180 to avoid odd behavior
        if rotation < -180:
            rotation = -180
        if rotation > 180:
            rotation = 180
    except Exception:
        rotation = 0

    # Watermark sources (image mode)
    wm_path = payload.get('watermarkPath') or opts.get('watermarkPath')
    wm_data_url = payload.get('watermarkDataUrl') or opts.get('watermarkDataUrl')

    # Images array might be [{url,name}, ...] or ["dataurl", ...]
    images = payload.get('images') or []
    norm_images = []
    if isinstance(images, list):
        for i, it in enumerate(images):
            if isinstance(it, dict):
                u = it.get('url') or it.get('dataUrl') or it.get('src')
                n = it.get('name') or f'watermarked_{i+1}.png'
            else:
                u = it
                n = f'watermarked_{i+1}.png'
            if u:
                norm_images.append({'url': u, 'name': n})

    return {
        'mode': mode,
        'text': text,
        'textSize': int(text_size),
        'textColor': text_color,
        'pos': pos,
        'padding': {'x': int(padx), 'xUnit': unit_x, 'y': int(pady), 'yUnit': unit_y},
        'scale': bool(scale),
        'opacity': float(opacity),
        'rotation': int(rotation),
        'watermarkPath': wm_path,
        'watermarkDataUrl': wm_data_url,
        'images': norm_images,
    }

# NEW: decode http(s) URLs or data/base64 into PIL
def _decode_any_to_pil(data: str) -> PILImage.Image:
    try:
        s = (data or '').strip()
        if s.startswith('http://') or s.startswith('https://'):
            # Avoid extra deps: use urllib
            import urllib.request
            with urllib.request.urlopen(s, timeout=10) as resp:
                raw = resp.read()
            return PILImage.open(io.BytesIO(raw)).convert('RGBA')
        # Fallback to data-url/base64 decoder
        return _decode_data_url_to_pil(s)
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

@app.route('/api/watermark/test', methods=['GET'])
def watermark_test():
    logger.info("[WATERMARK] /api/watermark/test OK")
    return jsonify({'ok': True, 'ts': int(time.time())})

# NEW: helpers for watermark API
def _decode_data_url_to_pil(data_url: str) -> PILImage.Image:
    """Decode a data URL (data:image/png;base64,...) into a PIL image"""
    try:
        if ',' in data_url:
            _, b64 = data_url.split(',', 1)
        else:
            b64 = data_url
        raw = base64.b64decode(b64)
        return PILImage.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def _pil_to_data_url(img: PILImage.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

# INVISIBLE watermark endpoint (DWT-DCT based)
@app.route('/api/watermark/apply-invisible', methods=['POST', 'OPTIONS'], endpoint='watermark_apply_invisible')
def apply_invisible_watermark():
    """Apply invisible watermark using DWT-DCT technique"""
    # CORS preflight
    if request.method == 'OPTIONS':
        resp = jsonify({'message': 'CORS preflight'})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        resp.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        resp.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return resp

    try:
        from invisible_watermark import apply_invisible_watermark as apply_inv_wm, extract_watermark_color, measure_ncc
    except ImportError:
        return jsonify({'error': 'Invisible watermarking not available. Install required packages: pywt, scipy, scikit-image'}), 500

    # Get current user if authenticated - FIXED
    current_user = get_user_from_request()

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({'error': 'Invalid JSON body'}), 400

    images = payload.get('images', [])
    watermark_mode = payload.get('watermarkMode', 'text')  # 'text' or 'image'
    watermark_text = payload.get('watermarkText', '')
    watermark_data_url = payload.get('watermarkDataUrl')

    # NEW: customization knobs from client
    alpha = payload.get('alpha', None)
    redundancy = payload.get('redundancy', None)

    # NEW: Create uploads directory for this session
    uploads_dir = Path(__file__).parent / 'uploads' / f'invisible_{uuid.uuid4().hex[:8]}'
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    out_images = []
    saved_paths = []
    
    for idx, item in enumerate(images):
        try:
            name = item.get('name') or f'image_{idx+1}.png'
            b64 = (item.get('url') or '').split(',', 1)[-1]
            host = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')

            wm_img = None
            if watermark_mode == 'image' and watermark_data_url:
                wm_b64 = watermark_data_url.split(',', 1)[-1]
                wm_img = Image.open(io.BytesIO(base64.b64decode(wm_b64)))

            result = apply_inv_wm(
                host_pil_image=host,
                watermark_mode=watermark_mode,
                watermark_text=watermark_text,
                watermark_pil_image=wm_img,
                alpha=alpha,
                redundancy=redundancy
            )
            
            # Get metadata from result
            wm_size = result.info.get('watermark_size')
            red = result.info.get('redundancy')
            wm_text = result.info.get('watermark_text', '')
            
            # CRITICAL: Save with ZERO compression via cv2 (preserves exact pixels)
            safe_name = name.replace('/', '_').replace('\\', '_')
            file_path = uploads_dir / safe_name
            
            # Step 1: Save pixel data with cv2 (lossless, no conversions)
            if hasattr(result, '_watermarked_array'):
                wm_bgr = cv2.cvtColor(result._watermarked_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(file_path), wm_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                logger.info("[INVISIBLE WM] Saved (cv2) %s", file_path)
            else:
                # Fallback: PIL save
                if hasattr(result, 'pnginfo') and result.pnginfo:
                    result.save(file_path, format='PNG', pnginfo=result.pnginfo, optimize=False, compress_level=0)
                else:
                    result.save(file_path, format='PNG', optimize=False, compress_level=0)
                logger.info("[INVISIBLE WM] Saved (PIL) %s", file_path)
            
            # Step 2: Add PNG text chunks WITHOUT re-encoding pixels
            try:
                import struct
                import zlib
                
                # Read the PNG file as binary
                with open(file_path, 'rb') as f:
                    png_data = bytearray(f.read())
                
                # PNG signature
                PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
                
                if not png_data.startswith(PNG_SIGNATURE):
                    raise ValueError("Not a valid PNG file")
                
                # Helper function to create PNG text chunk
                def create_text_chunk(keyword, text):
                    keyword_bytes = keyword.encode('latin-1')
                    text_bytes = text.encode('utf-8')
                    chunk_data = keyword_bytes + b'\x00' + text_bytes
                    
                    # Calculate CRC
                    chunk_type = b'tEXt'
                    crc = zlib.crc32(chunk_type + chunk_data) & 0xffffffff
                    
                    # Build chunk: length + type + data + CRC
                    chunk = struct.pack('>I', len(chunk_data)) + chunk_type + chunk_data + struct.pack('>I', crc)
                    return chunk
                
                # Create metadata chunks
                metadata_chunks = [
                    create_text_chunk('watermark_size', str(wm_size)),
                    create_text_chunk('redundancy', str(red)),
                    create_text_chunk('watermark_text', str(wm_text)),
                    create_text_chunk('alpha', str(result.info.get('alpha', alpha))),
                    create_text_chunk('imperceptibility_psnr', str(result.info.get('imperceptibility_psnr', '')))
                ]
                
                # Find IEND chunk (last 12 bytes: length(4) + type(4) + CRC(4))
                iend_pos = png_data.rfind(b'IEND')
                if iend_pos == -1:
                    raise ValueError("IEND chunk not found")
                
                # Insert metadata chunks BEFORE IEND
                insert_pos = iend_pos - 4  # Before the IEND length bytes
                for chunk in metadata_chunks:
                    png_data[insert_pos:insert_pos] = chunk
                
                # Write modified PNG
                with open(file_path, 'wb') as f:
                    f.write(png_data)
                
                logger.info("[INVISIBLE WM] Added PNG metadata")
            except Exception as meta_err:
                logger.warning("[INVISIBLE WM] Failed to add PNG metadata: %s", meta_err)
                # Don't fail the entire operation if metadata fails
            
            # VERIFICATION: Read back and test extraction
            try:
                verify_img = Image.open(file_path)
                verify_rgb = np.array(verify_img.convert('RGB'))
                
                # Extract watermark from saved file
                verify_extracted = extract_watermark_color(verify_rgb, wm_size, redundancy=red)
                
                # Compare with original reference map
                if hasattr(result, '_ref_map'):
                    verify_ncc = measure_ncc(result._ref_map, verify_extracted)
                elif hasattr(result, '_watermarked_array'):
                    mem_extracted = extract_watermark_color(result._watermarked_array, wm_size, redundancy=red)
                    verify_ncc = measure_ncc(mem_extracted, verify_extracted)
                else:
                    verify_ncc = None
                
                if verify_ncc is not None:
                    logger.info("[INVISIBLE WM] Post-save NCC=%.4f", verify_ncc)
                    
                    if verify_ncc < 0.9:
                        logger.warning("[INVISIBLE WM] Watermark degradation NCC=%.4f", verify_ncc)
                    else:
                        logger.info("[INVISIBLE WM] Watermark preserved NCC=%.4f", verify_ncc)
            except Exception as verify_err:
                logger.warning("[INVISIBLE WM] Post-save verification failed: %s", verify_err)
            
            saved_paths.append(file_path)
            
            logger.info("   Metadata: size=%s, red=%s, text='%s'", wm_size, red, wm_text)
            
            # Convert to base64 for frontend
            buf = io.BytesIO()
            result.save(buf, format='PNG', compress_level=0)
            out_images.append({ 
                'name': name, 
                'dataUrl': f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}",
                'serverPath': str(file_path)  # NEW: Include server path in response
            })
        except Exception as e:
            logger.error("[INVISIBLE WM] Failed at index %d: %s", idx, e)
            import traceback
            traceback.print_exc()
            return jsonify({ 'error': f'Invisible watermarking failed at index {idx}: {e}' }), 400

    logger.info("[INVISIBLE WM] Saved %d image(s) directory=%s", len(saved_paths), uploads_dir)
    return jsonify({ 'images': out_images, 'uploads_dir': str(uploads_dir) })

@app.route('/api/watermark/test-robustness', methods=['POST', 'OPTIONS'], endpoint='watermark_test_robustness')
def test_invisible_watermark_robustness():
    """Test robustness of invisible watermark against various attacks"""
    # CORS preflight
    if request.method == 'OPTIONS':
        resp = jsonify({'message': 'CORS preflight'})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        resp.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        resp.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return resp

    try:
        from invisible_watermark import test_watermark_robustness
    except ImportError:
        return jsonify({'error': 'Invisible watermarking not available. Install required packages: pywt, scipy, scikit-image'}), 500

    # Get current user if authenticated - FIXED
    current_user = get_user_from_request()

    try:
        payload = request.get_json(force=True) or {}
        images = payload.get('images', [])
        watermark_mode = payload.get('watermarkMode', 'text')
        watermark_text = payload.get('watermarkText', '')
        watermark_data_url = payload.get('watermarkDataUrl', '')
        # NEW: accept client-provided alpha/redundancy
        alpha = payload.get('alpha', None)
        redundancy = payload.get('redundancy', None)

        if not images:
            return jsonify({'error': 'No images provided'}), 400

        if watermark_mode not in ['text', 'image']:
            return jsonify({'error': 'Invalid watermark mode. Use "text" or "image".'}), 400

        logger.info("[ROBUSTNESS] Starting robustness test for %d image(s)", len(images))
        logger.info("   Mode: %s, Alpha: %s", watermark_mode, alpha)

        # Process only the first image for robustness testing
        first_image = images[0]
        name = first_image.get('name', 'test')
        # FIXED: Handle both 'dataUrl' and 'url' keys
        data_url = first_image.get('url') or first_image.get('dataUrl', '')

        if not data_url:
            return jsonify({'error': 'Image data missing'}), 400

        # Decode base64 image
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        img_bytes = base64.b64decode(data_url)
        img_pil = PILImage.open(io.BytesIO(img_bytes))

        # Prepare watermark
        watermark_pil = None
        if watermark_mode == 'image':
            if not watermark_data_url:
                return jsonify({'error': 'Watermark image is required for image mode'}), 400
            if ',' in watermark_data_url:
                watermark_data_url = watermark_data_url.split(',', 1)[1]
            wm_bytes = base64.b64decode(watermark_data_url)
            watermark_pil = PILImage.open(io.BytesIO(wm_bytes))
        else:  # text mode
            if not watermark_text:
                return jsonify({'error': 'Watermark text is required for text mode'}), 400

        # Now test robustness
        test_results = test_watermark_robustness(
            watermarked_pil_image=img_pil,
            watermark_mode=watermark_mode,
            watermark_text=watermark_text if watermark_mode == 'text' else None,
            watermark_pil_image=watermark_pil if watermark_mode == 'image' else None,
            alpha=alpha,
            redundancy=redundancy
        )

        logger.info("[ROBUSTNESS] Completed tests=%d", len(test_results['results']))

        return jsonify({
            'success': True,
            'results': test_results['results'],
            'original_image': test_results['original_image'],
            'watermark_size': test_results['watermark_size'],
            'redundancy': test_results['redundancy'],
            'alpha': test_results['alpha'],
            'imperceptibility_psnr': test_results.get('imperceptibility_psnr'),
            'image_name': name
        })
    except Exception as e:
        logger.error("[ROBUSTNESS] Failed: %s", e)
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Robustness testing failed: {e}'}), 500


# PRIMARY watermark endpoint
@app.route('/api/watermark/apply', methods=['POST', 'OPTIONS'], endpoint='watermark_apply_main')
def apply_watermark_main():
    """Apply watermark (image or text) to one or more images and return data URLs"""
    # CORS preflight
    if request.method == 'OPTIONS':
        resp = jsonify({'message': 'CORS preflight'})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        resp.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        resp.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return resp

    if not WATERMARK_AVAILABLE:
        return jsonify({'error': 'WaterMarker not available on server'}), 500
    
    # Get authenticated user (optional - anonymous watermarking still allowed)
    current_user = get_user_from_request()
    if current_user:
        logger.info("[WATERMARK] Auth user=%s", current_user.email)
    else:
        logger.info("[WATERMARK] Anonymous request")

    # Accept JSON; log raw body length for debugging
    try:
        payload = request.get_json(force=True) or {}
        # Note: removed verbose payload dump to avoid logging large base64 data
    except Exception as e:
        logger.error("[WATERMARK] Invalid JSON body: %s", e)
        return jsonify({'error': 'Invalid JSON body'}), 400

    # Normalize payload keys from various UIs
    norm = _normalize_watermark_payload(payload)

    images = norm.get('images') or []
    mode = norm.get('mode', 'image')
    pos = norm.get('pos', 'SE')
    padding = norm.get('padding') or {'x': 20, 'xUnit': 'px', 'y': 5, 'yUnit': 'px'}
    scale = bool(norm.get('scale', True))
    opacity = float(norm.get('opacity', 0.5))
    rotation = int(norm.get('rotation', 0))
    text = norm.get('text')
    text_size = int(norm.get('textSize', 32))
    text_color = norm.get('textColor') or '#FFFFFF'
    wm_path = norm.get('watermarkPath')
    wm_data_url = norm.get('watermarkDataUrl')

    logger.info("[WATERMARK] /api/watermark/apply called")
    logger.info("[WATERMARK] mode=%s pos=%s opacity=%.3f rotation=%d count=%d",
                mode, pos, opacity, rotation, len(images))

    if not isinstance(images, list) or len(images) == 0:
        return jsonify({'error': 'images array is required'}), 400

    if mode.lower() not in ('image', 'text'):
        return jsonify({'error': "mode must be 'image' or 'text'"}), 400

    # Prepare WaterMarker
    tmp_wm_file = None
    wm_init_path = None
    try:
        if mode.lower() == 'image':
            if wm_path and os.path.isfile(wm_path):
                wm_init_path = wm_path
            elif wm_data_url:
                # write to temp file
                if ',' in wm_data_url:
                    _, b64 = wm_data_url.split(',', 1)
                else:
                    b64 = wm_data_url
                raw = base64.b64decode(b64)
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                with open(tmp_path, 'wb') as f:
                    f.write(raw)
                tmp_wm_file = tmp_path
                wm_init_path = tmp_wm_file
            else:
                return jsonify({'error': 'watermarkPath or watermarkDataUrl required for image mode'}), 400
        wm = WaterMarker(wm_init_path, overwrite=True)
    except Exception as e:
        if tmp_wm_file:
            try: os.remove(tmp_wm_file)
            except: pass
        return jsonify({'error': f'Failed to initialize WaterMarker: {e}'}), 500

    # Convert padding dict to tuple format expected by WaterMarker
    try:
        padding_tuple = ((int(padding.get('x', 20)), str(padding.get('xUnit', 'px'))),
                         (int(padding.get('y', 5)), str(padding.get('yUnit', 'px'))))
    except Exception:
        return jsonify({'error': 'Invalid padding format'}), 400

    out_images = []
    try:
        for idx, item in enumerate(images):
            url = item.get('url')
            name = item.get('name') or f'watermarked_{idx+1}.png'
            if not url:
                return jsonify({'error': f'image at index {idx} missing url'}), 400

            try:
                src = _decode_any_to_pil(url)
            except Exception as e:
                return jsonify({'error': f'Invalid image at index {idx}: {e}'}), 400

            # Apply watermark in-memory with rotation support
            try:
                # If rotation is needed, we need to manually handle watermark application
                if rotation and rotation != 0:
                    # Prepare the watermark (image or text)
                    image = src.convert("RGBA")
                    
                    if mode.lower() == "text" and text:
                        # Create text watermark
                        watermark_copy = wm.create_text_watermark(text, text_size, text_color, opacity)
                    else:
                        # Image watermark
                        if scale and (not wm.previous_size or wm.previous_size != image.size):
                            watermark_copy = wm.scale_watermark(image)
                            needs_opacity = opacity < 1
                        else:
                            watermark_copy = (wm.watermark or PILImage.new("RGBA", (1, 1), (0, 0, 0, 0))).copy()
                            needs_opacity = opacity < 1
                        
                        wm.previous_size = image.size
                        
                        if needs_opacity and watermark_copy.mode != "RGBA":
                            watermark_copy = watermark_copy.convert("RGBA")
                        if needs_opacity:
                            watermark_copy = wm.change_opacity(watermark_copy, opacity)
                    
                    # Rotate the watermark (not the entire image!)
                    # PIL rotates counter-clockwise by default, so negate for clockwise rotation
                    watermark_copy = watermark_copy.rotate(-rotation, expand=True, fillcolor=(0, 0, 0, 0))
                    
                    # Get position and apply
                    x, y = wm.get_watermark_position(image, watermark_copy, pos=pos, padding=padding_tuple)
                    out_pil = image.copy()
                    try:
                        out_pil.paste(watermark_copy, box=(x, y), mask=watermark_copy)
                    except ValueError:
                        out_pil.paste(watermark_copy, box=(x, y))
                    out_pil = out_pil.convert("RGBA")
                else:
                    # No rotation - use standard method
                    out_pil = wm.apply_watermark_pil(
                        pil_image=src,
                        scale=scale,
                        pos=pos,
                        padding=padding_tuple,
                        opacity=opacity,
                        mode=mode.lower(),
                        text=(text if mode.lower() == 'text' else None),
                        text_size=text_size,
                        text_color=text_color
                    )
                    
            except Exception as e:
                return jsonify({'error': f'Watermarking failed at index {idx}: {e}'}), 500

            # Save to database if user is authenticated
            if current_user:
                try:
                    save_watermarked_image(
                        user_id=current_user.id,
                        original_path=name,
                        watermarked_pil=out_pil,
                        watermark_text=text if mode.lower() == 'text' else 'Image watermark',
                        position=pos,
                        opacity=opacity
                    )
                except Exception as e:
                    logger.warning("[WATERMARK] Save gallery failed: %s", e)
            
            out_images.append({
                'name': name,
                'dataUrl': _pil_to_data_url(out_pil, fmt='PNG')
            })
    finally:
        # Cleanup temp watermark
        if tmp_wm_file:
            try: os.remove(tmp_wm_file)
            except: pass

    if current_user and len(out_images) > 0:
        logger.info("[WATERMARK] Applied to %d image(s) saved for user=%s", len(out_images), current_user.email)
    else:
        logger.info("[WATERMARK] Applied to %d image(s)", len(out_images))
    return jsonify({'images': out_images})

# Aliases for compatibility
@app.route('/api/watermark', methods=['POST', 'OPTIONS'], endpoint='watermark_apply_alias')
def apply_watermark_alias():
    return apply_watermark_main()

@app.route('/watermark/apply', methods=['POST', 'OPTIONS'], endpoint='watermark_apply_compat')
def apply_watermark_compat():
    return apply_watermark_main()

# NEW: Alternative endpoint using watermarkdwt.py directly
@app.route('/api/watermark/apply-dwt', methods=['POST'])
def apply_watermark_dwt():
    """Alternative endpoint using watermarkdwt.py directly"""
    data = request.json
    
    try:
        # Save images to temp files
        temp_dir = tempfile.mkdtemp()
        input_paths = []
        
        for idx, img_data in enumerate(data['images']):
            # Decode base64
            img_bytes = base64.b64decode(img_data['url'].split(',')[1])
            temp_path = os.path.join(temp_dir, f'input_{idx}.png')
            with open(temp_path, 'wb') as f:
                f.write(img_bytes)
            input_paths.append(temp_path)
        
        # Save watermark if image mode
        if data['mode'] == 'image':
            wm_bytes = base64.b64decode(data['watermarkDataUrl'].split(',')[1])
            wm_path = os.path.join(temp_dir, 'watermark.png')
            with open(wm_path, 'wb') as f:
                f.write(wm_bytes)
        
        # Call watermarkdwt.py as subprocess
        script_path = r"D:\WatermarkGAN\Watermark-UI\DWT_DCT_Watermarking\watermarkdwt.py"
        
        for input_path in input_paths:
            cmd = [
                'python', script_path,
                '--image', input_path,
                '--watermark_mode', data['mode'],
                '--alpha', str(0.38)  # Your locked config
            ]
            
            if data['mode'] == 'text':
                cmd.extend(['--text', data['text']])
            else:
                cmd.extend(['--watermark', wm_path])
            
            subprocess.run(cmd, check=True)
        
        # Read results, encode as base64, return
        # ... cleanup temp files
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watermark/debug-extract-latest', methods=['GET'])
def debug_extract_latest():
    """Debug endpoint: Extract watermark from most recent uploaded image"""
    try:
        from invisible_watermark import extract_watermark_dwt_dct, rgb_to_ycbcr
        import os
        from pathlib import Path
        
        # Find most recent image
        uploads_dir = Path(__file__).parent / 'uploads'
        all_images = []
        for root, dirs, files in os.walk(uploads_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    full_path = Path(root) / file
                    all_images.append(full_path)
        
        if not all_images:
            return jsonify({'error': 'No images found in uploads'}), 404
        
        all_images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        img_path = all_images[0]
        
        img = Image.open(img_path)
        
        # NEW: Try reading PNG text chunks (where metadata is actually stored)
        wm_size = None
        redundancy = None
        watermark_text = None
        
        if hasattr(img, 'text'):
            # PNG text chunks (the REAL metadata)
            wm_size = int(img.text.get('watermark_size', 9))
            redundancy = int(img.text.get('redundancy', 3))
            watermark_text = img.text.get('watermark_text', '')
        
        # Fallback to .info dict (only works for in-memory images)
        if wm_size is None:
            wm_size = img.info.get('watermark_size', 9)
            redundancy = img.info.get('redundancy', 3)
            watermark_text = img.info.get('watermark_text', '')
        
        logger.info("[DEBUG] Metadata size=%s red=%s text='%s'", wm_size, redundancy, watermark_text)
        
        # Extract
        wm_rgb = np.array(img.convert('RGB'))
        Y, _, _ = rgb_to_ycbcr(wm_rgb)
        extracted = extract_watermark_dwt_dct(Y, wm_size, redundancy)
        
        density = float(np.mean(extracted > 0))
        
        return jsonify({
            'file': str(img_path),
            'metadata': {
                'watermark_size': wm_size,
                'redundancy': redundancy,
                'watermark_text': watermark_text
            },
            'extraction': {
                'shape': extracted.shape,
                'density': round(density, 4),
                'unique_values': [float(v) for v in np.unique(extracted)],
                'pattern_preview': extracted[:3, :3].tolist()
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    # Keep debug on for development, but disable the reloader to avoid parent process exit
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)



