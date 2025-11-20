from flask import Blueprint, request, jsonify
import os
import tempfile
import uuid
import base64
import io
from PIL import Image
import numpy as np

# Import your extraction functions directly (no subprocess needed!)
from invisible_watermark import extract_watermark_color, measure_ncc, create_text_watermark_hash, _prepare_capacity_and_wm, rgb_to_ycbcr

extract_bp = Blueprint('extract_bp', __name__)

@extract_bp.route('/api/extract-watermark', methods=['POST', 'OPTIONS'])
def extract_watermark_api():
    # CORS preflight
    if request.method == 'OPTIONS':
        resp = jsonify({'message': 'CORS preflight'})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        resp.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        resp.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return resp

    try:
        # Handle both form-data (file upload) and JSON (base64)
        claimed_text = None
        claimed_image_pil = None
        
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
            claimed_text = data.get('text', '')
            claimed_image_data = data.get('watermark_image')  # NEW
            
            if not image_data:
                return jsonify({'error': 'No image provided'}), 400
            
            # Decode base64 watermarked image
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            img_bytes = base64.b64decode(image_data)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # NEW: Decode claimed watermark image if provided
            if claimed_image_data:
                if ',' in claimed_image_data:
                    claimed_image_data = claimed_image_data.split(',', 1)[1]
                claimed_img_bytes = base64.b64decode(claimed_image_data)
                claimed_image_pil = Image.open(io.BytesIO(claimed_img_bytes))
        else:
            # Handle file upload
            if 'image' not in request.files:
                return jsonify({'error': 'No image uploaded'}), 400
            image_file = request.files['image']
            img_pil = Image.open(image_file).convert('RGB')
            claimed_text = request.form.get('text', '')
            
            # NEW: Handle watermark image upload
            if 'watermark_image' in request.files:
                claimed_image_pil = Image.open(request.files['watermark_image'])
        
        # Convert to numpy
        img_rgb = np.array(img_pil, dtype=np.uint8)
        
        # Try to read watermark metadata from PNG text chunks
        wm_size = None
        redundancy = None
        wm_text = ''
        
        if hasattr(img_pil, 'text'):
            try:
                wm_size = int(img_pil.text.get('watermark_size', 0))
                redundancy = int(img_pil.text.get('redundancy', 3))
                wm_text = img_pil.text.get('watermark_text', '')
                print(f"[EXTRACT] Metadata from PNG: wm_size={wm_size}, redundancy={redundancy}, text='{wm_text}'")
            except Exception as e:
                print(f"[EXTRACT] Failed to read PNG metadata: {e}")
        
        # Fallback - try common sizes
        if not wm_size or wm_size == 0:
            print("[EXTRACT] No metadata found, trying common sizes: 9, 16, 32")
            best_size = 9
            best_density = 0
            
            for test_size in [9, 16, 32]:
                try:
                    test_bits = extract_watermark_color(img_rgb, test_size, redundancy=redundancy or 3)
                    test_density = float(np.mean(test_bits > 0))
                    print(f"[EXTRACT] Try size={test_size}: density={test_density:.3f}")
                    
                    if abs(test_density - 0.5) < abs(best_density - 0.5):
                        best_size = test_size
                        best_density = test_density
                except Exception:
                    continue
            
            wm_size = best_size
            print(f"[EXTRACT] Auto-detected wm_size={wm_size}")
        
        if not redundancy:
            redundancy = 3
        
        # Extract watermark bits with detected/read size
        extracted_bits = extract_watermark_color(img_rgb, wm_size, redundancy=redundancy)
        
        # Calculate density
        density = float(np.mean(extracted_bits > 0))
        print(f"[EXTRACT] Extracted: size={wm_size}×{wm_size}, density={density:.3f}")
        
        # Convert extracted pattern to image for visualization
        bits_img = (extracted_bits > 0).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(bits_img, mode='L').save(buf, format='PNG')
        pattern_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Build EXTRACTION RESULTS
        extraction_results = {
            'watermark_size': wm_size,
            'redundancy': redundancy,
            'density': round(density, 4),
            'pattern_visualization': f'data:image/png;base64,{pattern_b64}',
            'metadata_text': wm_text if wm_text else None,
            'is_image_watermark': wm_text.startswith('IMG_SHA256:') if wm_text else False  # NEW
        }
        
        # Build VERIFICATION RESULTS
        verification_results = {
            'attempted': bool(claimed_text or claimed_image_pil),
            'claimed_text': claimed_text if claimed_text else None,
            'claimed_image': bool(claimed_image_pil),
            'result': 'not_attempted',
            'ncc_score': 0.0,
            'threshold': 0.6,
            'verified_text': None
        }
        
        # Verify if user provided claimed text OR image
        if claimed_text or claimed_image_pil:
            try:
                # Determine watermark mode from metadata
                is_image_wm = wm_text and wm_text.startswith('IMG_SHA256:')
                
                if claimed_image_pil:
                    # IMAGE VERIFICATION: Hash the claimed watermark image
                    print(f"[VERIFY] Verifying with uploaded watermark image")
                    
                    # Hash the claimed image (same as embedding)
                    wm_gray = claimed_image_pil.convert('L')
                    wm_gray = wm_gray.resize((256, 256), Image.BICUBIC)
                    wm_array = np.array(wm_gray).flatten()
                    
                    import hashlib
                    wm_bytes = wm_array.tobytes()
                    hash_obj = hashlib.sha256(wm_bytes)
                    hash_hex = hash_obj.hexdigest()
                    
                    # Convert hash to binary pattern
                    hash_bytes = bytes.fromhex(hash_hex)
                    bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
                    ref_wm_gray = bits.reshape(256, 1).astype(np.float32)
                    
                    print(f"[VERIFY] Claimed image SHA-256: {hash_hex[:32]}...")
                    
                elif claimed_text:
                    # TEXT VERIFICATION: Hash the claimed text
                    print(f"[VERIFY] Verifying with text: '{claimed_text}'")
                    
                    if is_image_wm:
                        verification_results['result'] = 'mode_mismatch'
                        verification_results['message'] = 'This is an image watermark. Please upload the watermark image.'
                        print("[VERIFY] Mode mismatch: image watermark vs text claim")
                        return jsonify({
                            'success': True,
                            'extraction': extraction_results,
                            'verification': verification_results
                        })
                    
                    ref_wm_gray = create_text_watermark_hash(claimed_text, 256)
                
                # Generate reference pattern
                Y, _, _ = rgb_to_ycbcr(img_rgb)
                ref_map, ref_size, _, _ = _prepare_capacity_and_wm(Y, ref_wm_gray, redundancy=redundancy)
                
                # Resize if needed
                if ref_size != wm_size:
                    from scipy.ndimage import zoom
                    zoom_factor = wm_size / ref_size
                    ref_map_resized = zoom(ref_map, zoom_factor, order=1)
                    ref_map = np.where(ref_map_resized >= 0, 1.0, -1.0)
                
                # Compute NCC
                ncc_score = measure_ncc(ref_map, extracted_bits)
                verification_results['ncc_score'] = round(ncc_score, 4)
                
                print(f"[VERIFY] NCC score: {ncc_score:.4f}")
                
                if ncc_score > 0.6:
                    verification_results['result'] = 'verified'
                    if claimed_text:
                        verification_results['verified_text'] = claimed_text
                    else:
                        verification_results['verified_text'] = 'Image watermark verified'
                elif ncc_score > 0.3:
                    verification_results['result'] = 'weak_match'
                else:
                    verification_results['result'] = 'mismatch'
                    
            except Exception as e:
                import traceback
                print(f"[VERIFY] Verification failed: {e}")
                traceback.print_exc()
                verification_results['result'] = 'error'
                verification_results['message'] = str(e)
        
        return jsonify({
            'success': True,
            'extraction': extraction_results,
            'verification': verification_results
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Extraction failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500
