import argparse
import sys
import os
import numpy as np
import pywt
from PIL import Image, ImageOps
import cv2
import importlib.util

# Load existing invisible_watermark.py dynamically (same directory)
MODULE_PATH = os.path.join(os.path.dirname(__file__), 'invisible_watermark.py')
spec = importlib.util.spec_from_file_location("invisible_wm_module", MODULE_PATH)
wm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wm_mod)

MODEL = wm_mod.MODEL
LEVEL = wm_mod.LEVEL
DEFAULT_REDUNDANCY = wm_mod.REDUNDANCY

def infer_watermark_size(Y_channel, redundancy):
    # FIXED: Match invisible_watermark.py's _prepare_capacity_and_wm logic exactly
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    blocks_h = LL.shape[0] // 8
    blocks_w = LL.shape[1] // 8
    cap_blocks = blocks_h * blocks_w
    
    red = int(redundancy)
    s = int(np.floor(np.sqrt(max(1, cap_blocks // red))))
    if s < 4:
        s = 4
        red = max(1, cap_blocks // (s * s))  # Recalculate redundancy if size was capped
    
    return s, red  # Return both size and adjusted redundancy

def extract_bits(Y_channel, wm_size, redundancy):
    return wm_mod.extract_watermark_dwt_dct(Y_channel, wm_size, redundancy=redundancy)

def bits_to_image(bits_2d, upscale=8):
    # bits (+1/-1) -> 255 / 0
    img_small = (bits_2d > 0).astype(np.uint8) * 255
    pil_small = Image.fromarray(img_small, mode='L')
    pil_up = pil_small.resize((img_small.shape[1]*upscale, img_small.shape[0]*upscale), Image.Resampling.NEAREST)
    # Improve OCR readability
    pil_up = ImageOps.autocontrast(pil_up)
    return pil_up

# NEW: Build reference map for IMAGE watermarks (mirrors embedding)
def _build_ref_map_from_image(image_path: str, size: int) -> np.ndarray:
    """
    Load original watermark image, convert to grayscale, resize to 256 then to (size,size),
    threshold using 40th percentile of non-zero pixels, and map to +1/-1.
    """
    try:
        img = Image.open(image_path).convert('L')
    except Exception:
        return None
    base_size = 256
    img_resized_large = img.resize((base_size, base_size), Image.Resampling.LANCZOS)
    wm_resized = img_resized_large.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(wm_resized, dtype=np.uint8)
    nz = arr[arr > 0]
    if nz.size >= 3:
        thr = np.percentile(nz, 40)
        bits01 = (arr >= thr).astype(np.uint8)
    else:
        bits01 = (arr > 128).astype(np.uint8)
    return bits01.astype(np.float64) * 2.0 - 1.0

# NEW: Render a clean grayscale source image for TEXT watermarks (256x256)
def _render_text_source_image(text: str, base_size: int = 256) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    img = Image.new('L', (base_size, base_size), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, base_size // 8))
    except:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except:
        tw, th = draw.textsize(text, font=font)
    x = (base_size - tw) // 2
    y = (base_size - th) // 2
    draw.text((x, y), text, fill=255, font=font)
    return img

# NEW: Improved text reconstruction using multiple upscaling strategies
def reconstruct_text_from_bits(bits_2d, meta_text=None):
    """
    Reconstruct text from extracted binary pattern using multiple strategies
    """
    size = bits_2d.shape[0]
    
    # Strategy 1: If we have metadata, verify it matches the extracted pattern
    if meta_text:
        template = create_text_watermark_template(meta_text, size)
        if template is not None:
            correlation = float(np.mean(bits_2d * template))
            score = (correlation + 1.0) / 2.0
            
            # If metadata text matches well (>50% correlation), trust it
            if score > 0.5:
                print(f"[METADATA MATCH] Text '{meta_text}' matches extracted pattern (score={score:.4f})")
                return meta_text, 'metadata_verified'
    
    # Strategy 2: Multi-scale OCR with different upscaling factors
    best_text = None
    best_confidence = 0
    
    for upscale_factor in [16, 24, 32]:
        try:
            # Create high-resolution version for OCR
            wm_img = bits_to_image(bits_2d, upscale=upscale_factor)
            
            # Try OCR
            text = ocr_extract_text(wm_img).strip()
            
            if text and len(text) >= 1:
                # Simple confidence: longer texts with alphanumeric chars are more likely correct
                confidence = len(text) * (1 if text.isalnum() else 0.5)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_text = text
                    
                print(f"[OCR @{upscale_factor}x] Extracted: '{text}' (confidence={confidence:.2f})")
        except Exception as e:
            print(f"[OCR @{upscale_factor}x] Failed: {e}")
    
    if best_text:
        return best_text, 'ocr_multiscale'
    
    # Strategy 3: Template matching as fallback (existing code)
    text_match, score = match_text_patterns(bits_2d)
    if text_match and score > 0.5:
        return text_match, 'template_match'
    
    # Strategy 4: Return pattern info as last resort
    density = float(np.mean(bits_2d > 0))
    return f"PATTERN(density={density:.3f})", 'heuristic'

def ocr_extract_text(pil_img):
    try:
        import pytesseract
        txt = pytesseract.image_to_string(
            pil_img,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@_-"
        )
        return txt.strip()
    except Exception:
        return ""

def naive_shape_guess(bits_2d):
    # Fallback: count foreground density and return a simple tag
    density = np.mean(bits_2d > 0)
    if density < 0.05:
        return "EMPTY"
    if density > 0.70:
        return "SOLID"
    # Attempt simple horizontal projection to guess characters count
    col_sum = np.sum(bits_2d > 0, axis=0)
    transitions = 0
    prev = col_sum[0] > 0
    for v in col_sum[1:]:
        cur = v > 0
        if cur != prev:
            transitions += 1
            prev = cur
    approx_chars = max(1, transitions // 2)
    return f"TEXT({approx_chars} chars est.)"

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img), img

def _entropy(bits_2d):
    p = float(np.mean(bits_2d > 0))
    p = max(min(p, 1 - 1e-9), 1e-9)
    return - (p * np.log2(p) + (1 - p) * np.log2(1 - p)), p

def _candidate_sizes(Y_channel, r, spread=3):
    base = infer_watermark_size(Y_channel, r)
    cand = sorted({max(4, base + k) for k in range(-spread, spread + 1)})
    return cand

def _auto_extract_best(Y_channel, redundancy_hint=None):
    # Build redundancy candidates (prioritize hint/default, then neighbors)
    default_r = int(redundancy_hint) if redundancy_hint else DEFAULT_REDUNDANCY
    r_list = [default_r, 2, 1, 4, 3, 5]  # Try hint first, then common values
    seen = set()
    candidates = []
    
    print(f"[DEBUG] Trying extraction with multiple (size, redundancy) combinations...")
    
    for r in r_list:
        if r in seen or r <= 0:
            continue
        seen.add(r)
        
        try:
            # FIXED: Use matching inference
            s, effective_r = infer_watermark_size(Y_channel, r)
            
            # Extract with inferred size and adjusted redundancy
            bits = extract_bits(Y_channel, s, effective_r)
            H, dens = _entropy(bits)
            
            # Score: prefer higher entropy, density near 0.5
            score = H - abs(dens - 0.5) * 0.05
            candidates.append((score, H, abs(dens - 0.5), dens, s, effective_r, bits))
            
            # NEW: Show all attempts
            print(f"  Candidate: size={s}, red={effective_r}, entropy={H:.3f}, density={dens:.3f}, score={score:.3f}")
        except Exception as e:
            print(f"[DEBUG] Failed r={r}: {e}")
            continue
    
    if not candidates:
        # Fallback: use default with best effort
        s, eff_r = infer_watermark_size(Y_channel, default_r)
        bits = extract_bits(Y_channel, s, eff_r)
        print(f"  [FALLBACK] size={s}, red={eff_r}")
        return bits, s, eff_r, float(np.mean(bits > 0))

    # Pick best by score (entropy primary), then density proximity to 0.5
    candidates.sort(key=lambda t: (t[0], t[1], -t[2]), reverse=True)
    best = candidates[0]
    _, best_H, _, best_dens, best_s, best_r, best_bits = best
    
    print(f"[DEBUG] Auto-selected: size={best_s}, redundancy={best_r}, entropy={best_H:.3f}, density={best_dens:.3f}")
    return best_bits, best_s, best_r, best_dens

# NEW: Add template generation for known texts
def create_text_watermark_template(text, size):
    """Create the EXACT template that was embedded (mirrors invisible_watermark.py)"""
    try:
        from PIL import ImageDraw, ImageFont
        font_size = max(12, size // 8)
        img = Image.new('L', (size*8, size*8), color=0)  # Create at higher res for better matching
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size*8)
        except:
            font = ImageFont.load_default()
        
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except:
            text_w, text_h = draw.textsize(text, font=font)
        
        x = (size*8 - text_w) // 2
        y = (size*8 - text_h) // 2
        draw.text((x, y), text, fill=255, font=font)
        
        # Resize to target size (matches _prepare_capacity_and_wm downsampling)
        img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.array(img_resized, dtype=np.float64)
        
        # Apply SAME binarization as _prepare_capacity_and_wm
        wm_u8 = np.clip(arr, 0, 255).astype(np.uint8)
        nz = wm_u8[wm_u8 > 0]
        if nz.size >= 3:
            thr_val = np.percentile(nz, 40)  # SAME 40th percentile
            bits01 = (wm_u8 >= thr_val).astype(np.uint8)
        else:
            bits01 = (wm_u8 > 128).astype(np.uint8)
        
        # Convert to +1/-1 (same as ref_map)
        ref_map = bits01.astype(np.float64) * 2.0 - 1.0
        return ref_map
    except Exception as e:
        print(f"[WARN] Template generation failed: {e}")
        return None

def match_text_patterns(extracted_bits, candidates=None):
    """Try to match extracted bits against common text patterns"""
    if candidates is None:
        # Common watermark texts to try
        candidates = [
            'Nandan', 'NANDAN', 'nandan',
            'Sample Watermark', 'SAMPLE', 'Sample',
            'Copyright', 'COPYRIGHT', '(c) Copyright',
            'Watermark', 'WATERMARK',
            'Protected', 'PROTECTED',
            'N', 'Na', 'Nan', 'Nand'  # Partial matches for 'Nandan'
        ]
    
    size = extracted_bits.shape[0]
    best_match = None
    best_score = -1
    
    print(f"\n[DEBUG] Trying template matching against {len(candidates)} candidates...")
    
    for text in candidates:
        try:
            template = create_text_watermark_template(text, size)
            if template is None:
                continue
            
            # Calculate correlation (NCC-like metric)
            # Both are +1/-1, so perfect match gives correlation = 1.0
            correlation = float(np.mean(extracted_bits * template))
            
            # Normalize to 0-1 range (correlation is in [-1, 1])
            score = (correlation + 1.0) / 2.0
            
            print(f"  - '{text}': correlation={correlation:.4f}, score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_match = text
        except Exception as e:
            print(f"  [WARN] Failed matching '{text}': {e}")
            continue
    
    # Consider it a match if correlation > 0.5 (means >75% bit agreement)
    if best_score > 0.5:
        print(f"\n[MATCH] Best match: '{best_match}' (score={best_score:.4f})")
        return best_match, best_score
    else:
        print(f"\n[NO MATCH] Best score was {best_score:.4f} for '{best_match}' (threshold=0.5)")
        return None, best_score

def smart_text_recovery(bits_2d, wm_img):
    """Multi-strategy text recovery"""
    # Strategy 1: OCR on upscaled image
    text_ocr = ocr_extract_text(wm_img).strip()
    if text_ocr and len(text_ocr) >= 3:  # Meaningful OCR result
        print(f"[OCR] Recovered text: '{text_ocr}'")
        return text_ocr, 'ocr'
    
    # Strategy 2: Template matching against common texts
    text_match, score = match_text_patterns(bits_2d)
    if text_match and score > 0.5:
        return text_match, 'template_match'
    
    # Strategy 3: Try inverted bits
    alt_bits = -bits_2d
    alt_img = bits_to_image(alt_bits, upscale=16)
    text_ocr_inv = ocr_extract_text(alt_img).strip()
    if text_ocr_inv and len(text_ocr_inv) >= 3:
        print(f"[OCR INVERTED] Recovered text: '{text_ocr_inv}'")
        return text_ocr_inv, 'ocr_inverted'
    
    text_match_inv, score_inv = match_text_patterns(alt_bits)
    if text_match_inv and score_inv > 0.5:
        return text_match_inv, 'template_match_inverted'
    
    # Strategy 4: Shape analysis fallback
    density = float(np.mean(bits_2d > 0))
    if density < 0.05:
        return "EMPTY", 'heuristic'
    elif density > 0.95:
        return "SOLID", 'heuristic'
    else:
        # Return pattern info
        return f"PATTERN(density={density:.3f})", 'heuristic'

def _build_ref_map_from_text(text: str, size: int) -> np.ndarray:
    """
    Reconstruct the reference +1/-1 map exactly like embedding:
    1. Render text into 256x256 grayscale canvas (same create_text_watermark behavior)
    2. Resize to (size,size)
    3. Threshold using 40th percentile of non-zero pixels
    """
    if not text:
        return None
    from PIL import ImageDraw, ImageFont
    base_size = 256
    img = Image.new('L', (base_size, base_size), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, base_size // 8))
    except:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except:
        tw, th = draw.textsize(text, font=font)
    x = (base_size - tw) // 2
    y = (base_size - th) // 2
    draw.text((x, y), text, fill=255, font=font)

    wm_resized = img.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(wm_resized, dtype=np.uint8)
    nz = arr[arr > 0]
    if nz.size >= 3:
        thr = np.percentile(nz, 40)
        bits01 = (arr >= thr).astype(np.uint8)
    else:
        # Degenerate fallback
        bits01 = (arr > 128).astype(np.uint8)
    return bits01.astype(np.float64) * 2.0 - 1.0  # 0->-1,1->+1

def _extract_bits_from_image(pil: Image.Image, size: int, redundancy: int) -> np.ndarray:
    rgb = np.array(pil.convert('RGB'), dtype=np.uint8)
    ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    Y = ycbcr[:, :, 0].astype(np.float64)
    return wm_mod.extract_watermark_dwt_dct(Y, size, redundancy=redundancy)

def _measure_ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.sign(a.flatten())
    bb = np.sign(b.flatten())
    aa[aa == 0] = 1
    bb[bb == 0] = 1
    return float(np.mean(aa * bb))

def extract_watermark(image_path: str, size_override=None, redundancy_override=None, known_text=None, watermark_image=None):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    
    # FIXED: Re-open image to ensure PNG text chunks are read (browser downloads issue)
    pil = Image.open(image_path)
    
    # Try reading metadata multiple ways
    meta_size = None
    meta_red = None
    meta_text = None
    
    # Method 1: Standard pil.text attribute
    if hasattr(pil, 'text') and pil.text:
        try:
            meta_size = int(pil.text.get('watermark_size', 0))
            meta_red = int(pil.text.get('redundancy', 0))
            meta_text = pil.text.get('watermark_text', '') or ''
            print(f"[METADATA] size={meta_size}, redundancy={meta_red}, text='{meta_text}'")
        except Exception as e:
            print(f"[METADATA] parse error: {e}")
    
    # Method 2: Fallback - re-read file and check info dict (for browser downloads)
    if not meta_size or not meta_red:
        try:
            with Image.open(image_path) as img_recheck:
                # Force load PNG chunks
                img_recheck.load()
                if hasattr(img_recheck, 'info') and img_recheck.info:
                    meta_size = int(img_recheck.info.get('watermark_size', 0))
                    meta_red = int(img_recheck.info.get('redundancy', 0))
                    meta_text = img_recheck.info.get('watermark_text', '') or ''
                    if meta_size and meta_red:
                        print(f"[METADATA FALLBACK] size={meta_size}, redundancy={meta_red}, text='{meta_text}'")
        except Exception as e:
            print(f"[METADATA FALLBACK] failed: {e}")

    # Method 3: Last resort - raw PNG chunk parsing (for browser downloads that strip metadata)
    if not meta_size or not meta_red:
        try:
            import struct
            with open(image_path, 'rb') as f:
                # Skip PNG signature (8 bytes)
                f.read(8)
                while True:
                    try:
                        # Read chunk length and type
                        length_data = f.read(4)
                        if len(length_data) < 4:
                            break
                        length = struct.unpack('>I', length_data)[0]
                        chunk_type = f.read(4).decode('latin1')
                        chunk_data = f.read(length)
                        f.read(4)  # Skip CRC
                        
                        # Look for tEXt chunks
                        if chunk_type == 'tEXt':
                            # tEXt format: keyword\0text
                            null_idx = chunk_data.find(b'\x00')
                            if null_idx > 0:
                                keyword = chunk_data[:null_idx].decode('latin1')
                                text = chunk_data[null_idx+1:].decode('latin1', errors='ignore')
                                
                                if keyword == 'watermark_size' and not meta_size:
                                    meta_size = int(text)
                                elif keyword == 'redundancy' and not meta_red:
                                    meta_red = int(text)
                                elif keyword == 'watermark_text' and not meta_text:
                                    meta_text = text
                                    
                                if meta_size and meta_red:
                                    print(f"[METADATA RAW CHUNKS] size={meta_size}, redundancy={meta_red}, text='{meta_text}'")
                                    break
                    except Exception:
                        break
        except Exception as e:
            print(f"[METADATA RAW CHUNKS] failed: {e}")

    # REPLACED FALLBACK BLOCK: unify behavior (use redundancy_override or DEFAULT_REDUNDANCY; no early return)
    if not meta_size or not meta_red:
        rgb = np.array(pil.convert('RGB'), dtype=np.uint8)
        ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        Y = ycbcr[:, :, 0].astype(np.float64)

        if redundancy_override is not None:
            forced_red = int(redundancy_override)
            print(f"[FALLBACK] Using user-provided redundancy={forced_red}, calculating size...")
        else:
            forced_red = int(DEFAULT_REDUNDANCY)
            print(f"[FALLBACK] No metadata found; using default redundancy={forced_red}")

        meta_size, meta_red = infer_watermark_size(Y, forced_red)
        meta_text = known_text or meta_text or ''
        print(f"[FALLBACK] Inferred watermark_size={meta_size}, effective_redundancy={meta_red}")

    # Direct inverse extraction
    bits = _extract_bits_from_image(pil, meta_size, meta_red)
    density = float(np.mean(bits > 0))
    print(f"[EXTRACT] bits shape={bits.shape}, density={density:.3f}")

    recovered_text = None
    recovery_method = 'none'
    verification_ncc = None
    watermark_type = 'unknown'

    # TEXT verification branch (unchanged logic) - prefer text if provided
    use_text_for_verify = meta_text or known_text
    if use_text_for_verify and not watermark_image:
        watermark_type = 'text'
        ref_map = _build_ref_map_from_text(use_text_for_verify, meta_size)
        if ref_map is not None and ref_map.shape == bits.shape:
            ncc = _measure_ncc(ref_map, bits)
            verification_ncc = ncc
            print(f"[VERIFY TEXT] NCC vs reference text map: {ncc:.4f}")
            if ncc >= 0.99:
                recovered_text = use_text_for_verify
                recovery_method = 'metadata_direct' if meta_text else 'override_direct'
            elif ncc >= 0.60:
                recovery_method = 'low_ncc_moderate'
                print(f"[AMBIGUOUS] NCC ({ncc:.4f}) between 0.60-0.99. Text may be incorrect.")
            else:
                recovery_method = 'low_ncc'
        else:
            recovery_method = 'ref_map_mismatch'

    # IMAGE verification branch (new)
    elif watermark_image:
        watermark_type = 'image'
        print(f"[IMAGE VERIFY] Using watermark image: {watermark_image}")
        ref_map_img = _build_ref_map_from_image(watermark_image, meta_size)
        if ref_map_img is None or ref_map_img.shape != bits.shape:
            print("[IMAGE VERIFY] Failed to build reference map (shape mismatch or load error).")
            recovery_method = 'image_ref_failed'
        else:
            ncc_img = _measure_ncc(ref_map_img, bits)
            verification_ncc = ncc_img
            print(f"[VERIFY IMAGE] NCC vs reference image map: {ncc_img:.4f}")
            if ncc_img >= 0.99:
                recovery_method = 'image_verified'
                recovered_text = 'IMAGE_MATCH'
            elif ncc_img >= 0.60:
                recovery_method = 'image_moderate'
                recovered_text = f'IMAGE_PARTIAL(NCC={ncc_img:.3f})'
                print(f"[AMBIGUOUS] Image NCC ({ncc_img:.4f}) moderate (0.60-0.99).")
            else:
                recovery_method = 'image_low_ncc'
                recovered_text = f'IMAGE_WEAK(NCC={ncc_img:.3f})'

    else:
        recovery_method = 'no_text_metadata'
        recovered_text = f"PATTERN(density={density:.3f})"

    if not recovered_text:
        recovered_text = f"PATTERN(density={density:.3f})"

    return {
        'image_path': image_path,
        'watermark_size': meta_size,
        'redundancy_used': meta_red,
        'extracted_text': recovered_text,
        'recovery_method': recovery_method,
        'foreground_density': density,
        'bits': bits,
        'watermark_type': watermark_type,
        'verification_ncc': verification_ncc
    }

def main():
    ap = argparse.ArgumentParser(
        description="Direct DWT-DCT invisible watermark extraction with text or image verification.",
        epilog="""
EXAMPLES:
  Auto-detect only:
    python wm_extract.py watermarked.png

  Verify text watermark:
    python wm_extract.py watermarked.png --text "Nandan"

  Verify image watermark:
    python wm_extract.py watermarked.png --wm-image original_logo.png

  Provide redundancy hint (improves size inference if metadata stripped):
    python wm_extract.py watermarked.png --wm-image logo.png --redundancy 3
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="Path to watermarked PNG")
    ap.add_argument("--out", "-o", help="Optional path to save reconstructed bit pattern PNG")
    ap.add_argument("--redundancy", "-r", type=int, help="Optional redundancy hint (default 3 if missing metadata)")
    ap.add_argument("--text", "-t", help="Known text watermark for verification")
    ap.add_argument("--wm-image", "-W", help="Path to original watermark image for verification (image mode)")
    args = ap.parse_args()

    if args.text and args.wm_image:
        print("[ERROR] Provide either --text OR --wm-image, not both.")
        sys.exit(1)

    try:
        result = extract_watermark(
            args.image,
            size_override=None,
            redundancy_override=args.redundancy,  # Works even if None (defaults internally to 3)
            known_text=args.text,
            watermark_image=args.wm_image
        )
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        sys.exit(1)

    if 'error' in result:
        print(f"[ERROR] {result['error']}")
        sys.exit(2)

    print("\n=== Watermark Extraction Report ===")
    print(f"Image: {result['image_path']}")
    print(f"Type: {result['watermark_type']}")
    print(f"Watermark size: {result['watermark_size']} x {result['watermark_size']}")
    print(f"Redundancy used: {result['redundancy_used']}")
    print(f"Recovered text / status: {result['extracted_text']}")
    print(f"Recovery method: {result['recovery_method']}")
    if result['verification_ncc'] is not None:
        print(f"Verification NCC: {result['verification_ncc']:.4f}")
    print(f"Foreground density: {result['foreground_density']:.4f}")

    if args.out:
        try:
            bits = result['bits']
            vis = (bits > 0).astype(np.uint8) * 255
            Image.fromarray(vis, mode='L').save(args.out, format='PNG')
            print(f"Reconstructed watermark pattern saved to: {args.out}")
        except Exception as e:
            print(f"[WARN] Failed to save pattern image: {e}")

if __name__ == "__main__":
    main()
