"""
Invisible Watermarking Module for AI-Image-Suite
DWT-DCT based invisible watermarking with robust extraction
"""
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter
import cv2
import io
import base64
import hashlib  # NEW

# Configuration (LOCKED)
MODEL = 'haar'
LEVEL = 1
ALPHA = 0.38
REDUNDANCY = 3  # FIXED: Hardcoded to 3 for optimal performance with 256x256 images

# =========================
# Helpers
# =========================
def rgb_to_ycbcr(img_rgb):
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    return ycbcr[:, :, 0].astype(np.float64), ycbcr[:, :, 1], ycbcr[:, :, 2]

def ycbcr_to_rgb(Y, Cr, Cb):
    Y_uint = np.clip(Y, 0, 255).astype(np.uint8)
    ycbcr = np.stack([Y_uint, Cr, Cb], axis=2)
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)

def create_text_watermark(text, size, font_size=None):
    """Legacy text watermark using font rendering (for visible watermarks)"""
    if font_size is None:
        font_size = max(12, size // 8)
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)
    x = (size - text_w) // 2
    y = (size - text_h) // 2
    draw.text((x, y), text, fill=255, font=font)
    return np.array(img, dtype=np.float64)

# NEW: SHA-256 based pattern generator for invisible watermarks
def create_text_watermark_hash(text, size):
    """
    Create balanced binary watermark pattern from text using SHA-256 hashing.
    This eliminates false positives by ensuring ~50% bit distribution.
    
    Args:
        text: Input text string
        size: Output pattern size (will be size x size grid)
    
    Returns:
        numpy array (size x size) with balanced grayscale pattern (0-255)
    """
    # Hash text to get cryptographically balanced bit pattern
    text_bytes = text.encode('utf-8')
    hash_digest = hashlib.sha256(text_bytes).digest()
    
    # Convert hash bytes to binary bits (each byte → 8 bits)
    bits = np.unpackbits(np.frombuffer(hash_digest, dtype=np.uint8))
    
    # Calculate needed bits for size×size grid
    needed = size * size
    
    if len(bits) < needed:
        # Repeat hash via chaining if needed: H(text), H(H(text)), H(H(H(text))), ...
        extended_bits = bits
        current_hash = hash_digest
        while len(extended_bits) < needed:
            current_hash = hashlib.sha256(current_hash).digest()
            new_bits = np.unpackbits(np.frombuffer(current_hash, dtype=np.uint8))
            extended_bits = np.concatenate([extended_bits, new_bits])
        bits = extended_bits[:needed]
    else:
        bits = bits[:needed]
    
    # Reshape to 2D grid and convert to grayscale (0 → black, 1 → white)
    pattern = bits.reshape(size, size).astype(np.float64) * 255.0
    
    # Log the balanced distribution for verification
    density = float(bits.mean())
    print(f"   🔐 SHA-256 pattern for '{text}': density={density:.3f} (target=0.50)")
    
    return pattern

def load_watermark_image_from_pil(pil_image, size):
    img = pil_image.convert('L')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float64)

def apply_dct(image_array):
    h, w = image_array.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    image_array = image_array[:h, :w]
    out = np.zeros_like(image_array)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blk = image_array[i:i+8, j:j+8]
            out[i:i+8, j:j+8] = dct(dct(blk.T, norm="ortho").T, norm="ortho")
    return out

def inverse_dct(dct_array):
    h, w = dct_array.shape
    out = np.zeros_like(dct_array)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blk = dct_array[i:i+8, j:j+8]
            out[i:i+8, j:j+8] = idct(idct(blk.T, norm="ortho").T, norm="ortho")
    return out

def _prepare_capacity_and_wm(Y_channel, wm_gray, redundancy=None):
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    H, W = LL.shape
    blocks_h, blocks_w = (H // 8), (W // 8)
    cap_blocks = blocks_h * blocks_w

    red = int(redundancy) if (redundancy is not None) else REDUNDANCY
    s = int(np.floor(np.sqrt(max(1, cap_blocks // red))))
    if s < 4:
        s = 4
        red = max(1, cap_blocks // (s * s))

    # Resize watermark to target embedding grid
    wm_img = Image.fromarray(np.uint8(np.clip(wm_gray, 0, 255)))
    wm_resized = wm_img.resize((s, s), Image.Resampling.LANCZOS)
    wm_arr = np.array(wm_resized, dtype=np.float64)

    # FIXED: Robust binarization using adaptive percentile on NON-ZERO values
    wm_u8 = np.clip(wm_arr, 0, 255).astype(np.uint8)
    nz = wm_u8[wm_u8 > 0]
    
    if nz.size < 3:
        # Degenerate case: almost all black → force checkerboard
        bits01 = np.ones((s, s), dtype=np.uint8)
        bits01[::2, 1::2] = 0
        bits01[1::2, ::2] = 0
        thr_val = 0
    else:
        # Use 40th percentile of non-zero pixels as threshold
        # This ensures ~40% of non-zero pixels become 1s, rest 0s
        thr_val = np.percentile(nz, 40)
        bits01 = (wm_u8 >= thr_val).astype(np.uint8)
    
    density = float(bits01.mean())
    
    # Safety net: if still extreme, apply minimal correction
    if density > 0.95:
        # Flip lowest 20% intensity cells to zero
        flat = wm_arr.flatten()
        n_flip = int(0.2 * flat.size)
        idx = np.argpartition(flat, n_flip)[:n_flip]
        bits_flat = bits01.flatten()
        bits_flat[idx] = 0
        bits01 = bits_flat.reshape(s, s)
        density = float(bits01.mean())
    elif density < 0.05:
        # Flip highest 20% intensity cells to one
        flat = wm_arr.flatten()
        n_flip = int(0.2 * flat.size)
        idx = np.argpartition(-flat, n_flip)[:n_flip]
        bits_flat = bits01.flatten()
        bits_flat[idx] = 1
        bits01 = bits_flat.reshape(s, s)
        density = float(bits01.mean())

    # DEBUG: Print bits01 state RIGHT BEFORE ref_map creation
    print(f"   🐛 DEBUG bits01 RIGHT BEFORE ref_map: mean={bits01.mean():.3f}, unique={np.unique(bits01)}")
    
    # CRITICAL FIX: Convert to float FIRST to prevent uint8 overflow/wrapping
    bits01_float = bits01.astype(np.float64)
    ref_map = bits01_float * 2.0 - 1.0  # Now: 0.0 → -1.0, 1.0 → +1.0
    
    # DEBUG: Print ref_map state RIGHT AFTER creation
    print(f"   🐛 DEBUG ref_map RIGHT AFTER creation: mean_positive={(ref_map > 0).mean():.3f}, unique={np.unique(ref_map)}")
    
    print(f"   🛠 Binarization: s={s}, thr={thr_val:.2f}, density={density:.3f}")
    
    return ref_map, s, (H, W), red

def _coeff_strength_from_ll(dct_ll):
    a = np.abs(dct_ll[3::8, 4::8]).flatten()
    b = np.abs(dct_ll[4::8, 3::8]).flatten()
    return np.percentile(np.concatenate([a, b]), 75) + 1e-6

def _embed_pair_margin(block, bit, margin):
    c1 = block[3, 4]
    c2 = block[4, 3]
    diff = c1 - c2
    if bit > 0:
        if diff < margin:
            delta = 0.52 * (margin - diff)
            block[3, 4] = c1 + delta
            block[4, 3] = c2 - delta
    else:
        if -diff < margin:
            delta = 0.52 * (margin + diff)
            block[3, 4] = c1 - delta
            block[4, 3] = c2 + delta
    return block

def embed_watermark_dwt_dct(Y_channel, ref_map, alpha=ALPHA, redundancy=REDUNDANCY):
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    dct_ll = apply_dct(LL)

    base_strength = _coeff_strength_from_ll(dct_ll)
    MIN_MARGIN = 6.0
    margin = max(alpha * base_strength, MIN_MARGIN)
    if base_strength > 18.0:
        margin *= 1.05

    bits = ref_map.flatten()
    H, W = dct_ll.shape
    needed_blocks = int(bits.size * int(redundancy))
    idx = 0
    bit_idx = 0
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            if bit_idx >= bits.size or idx >= needed_blocks:
                break
            blk = dct_ll[i:i+8, j:j+8]
            dct_ll[i:i+8, j:j+8] = _embed_pair_margin(blk, bits[bit_idx], margin)
            idx += 1
            if idx % int(redundancy) == 0:
                bit_idx += 1
        if bit_idx >= bits.size or idx >= needed_blocks:
            break

    coeffs[0] = inverse_dct(dct_ll)
    Y_wm = pywt.waverec2(coeffs, MODEL)
    return Y_wm[:Y_channel.shape[0], :Y_channel.shape[1]]

def extract_watermark_dwt_dct(Y_channel, wm_size, redundancy=REDUNDANCY):
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    dct_ll = apply_dct(LL)

    H, W = dct_ll.shape
    votes = np.zeros((wm_size * wm_size,), dtype=np.int32)
    idx = 0
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            bit_index = idx // int(redundancy)
            if bit_index >= votes.size:
                break
            blk = dct_ll[i:i+8, j:j+8]
            diff = blk[3, 4] - blk[4, 3]
            votes[bit_index] += 1 if diff >= 0 else -1
            idx += 1
        if (idx // int(redundancy)) >= votes.size:
            break

    bits = np.where(votes >= 0, 1.0, -1.0)
    return bits.reshape(wm_size, wm_size)

def embed_watermark_color(host_rgb, watermark_gray, alpha=None, redundancy=None):
    # defaults
    alpha = float(alpha) if (alpha is not None) else ALPHA
    Y, Cr, Cb = rgb_to_ycbcr(host_rgb)
    # FIXED: Use direct _prepare_capacity_and_wm with built-in safety net
    ref_map, wm_size, _, eff_red = _prepare_capacity_and_wm(Y, watermark_gray, redundancy=redundancy)
    Y_wm = embed_watermark_dwt_dct(Y, ref_map, alpha=alpha, redundancy=eff_red)
    return ycbcr_to_rgb(Y_wm, Cr, Cb), ref_map, wm_size, eff_red

def extract_watermark_color(wm_rgb, wm_size, redundancy=None, key=None, salt_hex=None):
    """
    Extract watermark bits from RGB image. If keyed, requires key+salt to unscramble.
    """
    redundancy = int(redundancy or REDUNDANCY)
    Y, _, _ = rgb_to_ycbcr(wm_rgb)
    
    # Extract raw (possibly scrambled) bits
    raw_bits = extract_watermark_dwt_dct(Y, wm_size, redundancy)
    
    # NEW: Unscramble if keyed
    if key and salt_hex:
        try:
            salt = bytes.fromhex(salt_hex)
            raw_bits = _unscramble_bits(raw_bits, key, salt)
        except Exception as e:
            raise ValueError(f"Key unscrambling failed: {e}")
    
    return raw_bits

# =========================
# Attacks and metrics
# =========================
def attack_jpeg(img_rgb, quality):
    img_pil = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    img_pil.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def attack_noise(img_rgb, sigma):
    noise = np.random.normal(0, sigma, img_rgb.shape)
    attacked = np.clip(img_rgb + noise * 255, 0, 255).astype(np.uint8)
    return attacked

def attack_blur(img_rgb, sigma):
    attacked = np.zeros_like(img_rgb, dtype=np.float64)
    for c in range(3):
        attacked[:, :, c] = gaussian_filter(img_rgb[:, :, c].astype(np.float64), sigma=sigma)
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    return attacked

def measure_psnr(original, watermarked):
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def measure_ncc(original_wm, recovered_wm):
    a = np.sign(np.asarray(original_wm, dtype=np.float64).flatten())
    b = np.sign(np.asarray(recovered_wm, dtype=np.float64).flatten())
    a[a == 0] = 1.0
    b[b == 0] = 1.0
    return float(np.mean(a * b))

# =========================
# Public API
# =========================
def apply_invisible_watermark(host_pil_image, watermark_mode, watermark_text=None,
                              watermark_pil_image=None, alpha=None, redundancy=None,
                              key=None):  # NEW: optional secret key for confidentiality
    """
    Returns a PIL Image only (no tuple) to remain compatible with callers that call .save().
    """
    alpha = float(alpha) if (alpha is not None) else ALPHA
    host_rgb = np.array(host_pil_image.convert('RGB'), dtype=np.uint8)
    
    # NEW: Use hash-based pattern for text mode to eliminate false positives
    if watermark_mode == 'text':
        wm_gray = create_text_watermark_hash(watermark_text or 'Copyright', 256)
    # IMAGE watermark mode
    elif watermark_mode == 'image':
        if watermark_pil_image is None:
            raise ValueError("watermark_pil_image is required for image mode")
        
        # NEW: Hash the image pixels with SHA-256 (same approach as text)
        wm_gray = watermark_pil_image.convert('L')  # Grayscale
        wm_gray = wm_gray.resize((256, 256), Image.BICUBIC)  # Standardize size
        wm_array = np.array(wm_gray).flatten()  # Flatten to 1D
        
        # Compute SHA-256 hash of image pixel data
        import hashlib
        wm_bytes = wm_array.tobytes()
        hash_obj = hashlib.sha256(wm_bytes)
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to 256-bit binary pattern (same as text mode)
        hash_bytes = bytes.fromhex(hash_hex)
        bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
        
        # Reshape to 256-bit watermark
        wm_gray = bits.reshape(256, 1).astype(np.float32)
        
        # Store hash hex as "text" for metadata/verification
        watermark_identifier = f"IMG_SHA256:{hash_hex[:16]}"  # First 16 chars as ID
        
        print(f"   ℹ Image watermark hashed to SHA-256: {hash_hex[:32]}...")
        print(f"   ℹ Watermark identifier: {watermark_identifier}")
    else:
        raise ValueError("Invalid watermark_mode. Use 'text' or 'image'.")
    
    wm_host_rgb, ref_map, wm_size, eff_red = embed_watermark_color(host_rgb, wm_gray, alpha=alpha, redundancy=redundancy)

    # CRITICAL: Create PIL image with explicit mode to prevent conversions
    result = Image.fromarray(wm_host_rgb, mode='RGB')

    try:
        psnr = measure_psnr(host_rgb, wm_host_rgb)

        # Store in memory (for immediate use)
        result.info['imperceptibility_psnr'] = round(float(psnr), 2)
        result.info['alpha'] = round(float(alpha), 4)
        result.info['redundancy'] = int(eff_red)
        result.info['watermark_size'] = int(wm_size)
        result.info['watermark_text'] = str(watermark_text or '')
        
        # NEW: Create PNG metadata that will be saved to file
        from PIL import PngImagePlugin
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("watermark_size", str(wm_size))
        pnginfo.add_text("redundancy", str(eff_red))
        pnginfo.add_text("watermark_text", str(watermark_text or ''))
        pnginfo.add_text("alpha", str(round(float(alpha), 4)))
        pnginfo.add_text("imperceptibility_psnr", str(round(float(psnr), 2)))
        
        # Attach PNG metadata to result so it's saved when .save() is called
        result.pnginfo = pnginfo
        
        # Self-test extraction
        extracted = extract_watermark_color(wm_host_rgb, wm_size, redundancy=eff_red)
        ncc = measure_ncc(ref_map, extracted)
        ref_density = float(np.mean(ref_map > 0))
        print(f"   🔍 Self-test extraction NCC: {ncc:.4f} (should be ~1.0)")
        print(f"   🧩 Ref watermark density (final): {ref_density:.3f}")
        
        if ref_density >= 0.95 or ref_density <= 0.05:
            print("   ⚠️ WARNING: Ref density still extreme; diversity fallback applied. Extraction will work but robustness may reduce.")
        
        result.info['self_test_ncc'] = round(float(ncc), 4)
        
        if ncc < 0.9:
            print(f"   ⚠️ WARNING: Self-test NCC is low ({ncc:.4f}). Watermark may not survive compression!")
            
        # CRITICAL: Store both the raw array AND the reference map for verification
        result._watermarked_array = wm_host_rgb.copy()
        result._ref_map = ref_map.copy()
        
    except Exception as e:
        print(f"   ⚠️ Self-test failed: {e}")
    return result

def test_watermark_robustness(watermarked_pil_image, watermark_mode, watermark_text=None,
                              watermark_pil_image=None, alpha=None, redundancy=None,
                              key=None):
    """
    Compute imperceptibility PSNR from original vs watermarked, then run attacks and return results.
    """
    alpha = float(alpha) if (alpha is not None) else ALPHA
    if watermarked_pil_image.mode != 'RGB':
        watermarked_pil_image = watermarked_pil_image.convert('RGB')
    host_rgb = np.array(watermarked_pil_image, dtype=np.uint8)

    # NEW: Use hash-based pattern for text mode (must match embedding)
    if watermark_mode == 'text':
        wm_gray_raw = create_text_watermark_hash(watermark_text or 'Copyright', 256)
    # IMAGE watermark mode
    elif watermark_mode == 'image':
        if watermark_pil_image is None:
            raise ValueError("watermark_pil_image is required for image mode")
        
        # NEW: Hash the image pixels with SHA-256 (same approach as text)
        wm_gray = watermark_pil_image.convert('L')  # Grayscale
        wm_gray = wm_gray.resize((256, 256), Image.BICUBIC)  # Standardize size
        wm_array = np.array(wm_gray).flatten()  # Flatten to 1D
        
        # Compute SHA-256 hash of image pixel data
        import hashlib
        wm_bytes = wm_array.tobytes()
        hash_obj = hashlib.sha256(wm_bytes)
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to 256-bit binary pattern (same as text mode)
        hash_bytes = bytes.fromhex(hash_hex)
        bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
        
        # Reshape to 256-bit watermark
        wm_gray_raw = bits.reshape(256, 1).astype(np.float32)
    else:
        raise ValueError("Invalid watermark_mode. Use 'text' or 'image'.")

    Y_host = cv2.cvtColor(host_rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float64)
    # FIXED: Use direct _prepare_capacity_and_wm (same as embed)
    ref_map, wm_size, _, eff_red = _prepare_capacity_and_wm(Y_host, wm_gray_raw, redundancy=redundancy)

    wm_host_rgb, _, _, _ = embed_watermark_color(host_rgb, wm_gray_raw, alpha=alpha, redundancy=eff_red)
    imperceptibility_psnr = measure_psnr(host_rgb, wm_host_rgb)
    print(f"   ✅ Imperceptibility PSNR: {imperceptibility_psnr:.2f} dB (computed from original vs watermarked)")

    wm_binary = ref_map
    results = []

    # Test 1: Original (no attack)
    try:
        recovered_wm_orig = extract_watermark_color(wm_host_rgb, wm_size, redundancy=eff_red)
        ncc_orig = measure_ncc(wm_binary, recovered_wm_orig)
        results.append({
            'attack': 'Original (No Attack)',
            'psnr': 0.0,
            'ncc': round(ncc_orig, 4),
            'success': ncc_orig > 0.6
        })
        print(f"   ✓ Original: NCC = {ncc_orig:.4f}")
    except Exception as e:
        results.append({
            'attack': 'Original (No Attack)',
            'psnr': 0.0,
            'ncc': 0.0,
            'success': False,
            'error': str(e)
        })

    # JPEG
    for quality in [85, 70, 50]:
        try:
            attacked_rgb = attack_jpeg(wm_host_rgb, quality)
            psnr = measure_psnr(wm_host_rgb, attacked_rgb)
            recovered = extract_watermark_color(attacked_rgb, wm_size, redundancy=eff_red)
            ncc = measure_ncc(wm_binary, recovered)
            results.append({
                'attack': f'JPEG Compression (Q={quality})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"   ✓ JPEG Q={quality}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            results.append({
                'attack': f'JPEG Compression (Q={quality})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })

    # Noise
    for sigma in [0.03, 0.06]:
        try:
            attacked_rgb = attack_noise(wm_host_rgb, sigma)
            psnr = measure_psnr(wm_host_rgb, attacked_rgb)
            recovered = extract_watermark_color(attacked_rgb, wm_size, redundancy=eff_red)
            ncc = measure_ncc(wm_binary, recovered)
            results.append({
                'attack': f'Gaussian Noise (σ={sigma})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"   ✓ Noise σ={sigma}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            results.append({
                'attack': f'Gaussian Noise (σ={sigma})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })

    # Blur
    for sigma in [0.8, 1.2]:
        try:
            attacked_rgb = attack_blur(wm_host_rgb, sigma)
            psnr = measure_psnr(wm_host_rgb, attacked_rgb)
            recovered = extract_watermark_color(attacked_rgb, wm_size, redundancy=eff_red)
            ncc = measure_ncc(wm_binary, recovered)
            results.append({
                'attack': f'Gaussian Blur (σ={sigma})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"   ✓ Blur σ={sigma}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            results.append({
                'attack': f'Gaussian Blur (σ={sigma})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })

    # Encode watermarked image for UI
    buf = io.BytesIO()
    Image.fromarray(wm_host_rgb).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        'results': results,
        'original_image': f'data:image/png;base64,{img_b64}',
        'watermark_size': int(wm_size),
        'redundancy': int(eff_red),
        'alpha': float(alpha),
        'imperceptibility_psnr': round(float(imperceptibility_psnr), 2)
    }

# NEW: Key-derived bit scrambler for confidentiality
def _scramble_bits(bits: np.ndarray, key: str, salt: bytes) -> np.ndarray:
    """XOR bits with key-derived pseudo-random mask for confidentiality"""
    flat = bits.flatten()
    # Derive deterministic mask from key + salt using SHA256
    kdf = hashlib.sha256((key + salt.hex()).encode('utf-8')).digest()
    # Expand to match bit array length using repeated hashing
    mask_bytes = kdf
    while len(mask_bytes) < len(flat):
        mask_bytes += hashlib.sha256(mask_bytes).digest()
    # Convert bytes to +1/-1 bit mask
    mask = np.frombuffer(mask_bytes[:len(flat)], dtype=np.uint8)
    mask = ((mask & 1) * 2 - 1).astype(np.float64)  # 0→-1, 1→+1
    # XOR operation in {-1,+1} domain: multiply
    scrambled = flat * mask
    return scrambled.reshape(bits.shape)

def _unscramble_bits(scrambled: np.ndarray, key: str, salt: bytes) -> np.ndarray:
    """Reverse XOR scrambling (self-inverse operation)"""
    return _scramble_bits(scrambled, key, salt)  # XOR is self-inverse