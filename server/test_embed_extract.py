import sys
sys.path.insert(0, 'D:\\WatermarkGAN\\AI-Image-Suite\\server')
from invisible_watermark import *

# Load test image
from PIL import Image
img = Image.open(r"C:\Users\nanda\Downloads\generated-1762582081151.png")

# Embed
print("🔐 Embedding watermark 'Nandan'...")
wm_img = apply_invisible_watermark(img, 'text', watermark_text='Nandan', alpha=0.35, redundancy=3)
print(f"   Self-test NCC: {wm_img.info.get('self_test_ncc')}")

# Extract immediately (no file save)
print("\n🔍 Extracting watermark...")
wm_rgb = np.array(wm_img.convert('RGB'))
Y, _, _ = rgb_to_ycbcr(wm_rgb)
wm_size = wm_img.info.get('watermark_size', 9)
red = wm_img.info.get('redundancy', 3)
extracted = extract_watermark_dwt_dct(Y, wm_size, red)

# Compute NCC
ref_map = create_text_watermark('Nandan', 256)
coeffs = pywt.wavedec2(Y, MODEL, level=LEVEL)
LL = coeffs[0]
ref_map_resized, _, _, _ = _prepare_capacity_and_wm(Y, ref_map, redundancy=red)
ncc = measure_ncc(ref_map_resized, extracted)

print(f"   Extraction NCC: {ncc:.4f}")
print(f"   Extracted bits density: {np.mean(extracted > 0):.3f}")
