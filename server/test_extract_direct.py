import sys
sys.path.insert(0, r'D:\WatermarkGAN\AI-Image-Suite\server')

from invisible_watermark import *
from PIL import Image
import os
from pathlib import Path

# Find the most recent watermarked image in uploads folder
uploads_dir = Path(r'D:\WatermarkGAN\AI-Image-Suite\server\uploads')

# UPDATED: Search in subdirectories too (invisible_*)

# Find all PNG files in uploads directory and subdirectories
all_images = []
for root, dirs, files in os.walk(uploads_dir):
    for file in files:
        # FILTER: Only PNG files that look like watermarked images
        if file.lower().endswith('.png') and ('watermarked' in file.lower() or 'generated' in file.lower() or 'image' in file.lower()):
            full_path = Path(root) / file
            all_images.append(full_path)

if not all_images:
    print("❌ No watermarked images found in uploads folder!")
    print(f"   Checked: {uploads_dir}")
    print(f"   Subdirectories: {list(uploads_dir.iterdir()) if uploads_dir.exists() else 'folder does not exist'}")
    sys.exit(1)

# Sort by modification time (most recent first)
all_images.sort(key=lambda p: p.stat().st_mtime, reverse=True)

# Use the most recent image
img_path = all_images[0]
print(f"📂 Using most recent image: {img_path}")
print(f"   Modified: {os.path.getmtime(img_path)}")

img = Image.open(img_path)

# NEW: Try reading PNG text chunks (where metadata is REALLY stored)
wm_size = None
redundancy = None
watermark_text = None

if hasattr(img, 'text'):
    # PNG text chunks contain the actual saved metadata
    wm_size = int(img.text.get('watermark_size', 9))
    redundancy = int(img.text.get('redundancy', 3))
    watermark_text = img.text.get('watermark_text', '')
    print(f"✅ Read metadata from PNG text chunks")

# Fallback to .info dict
if wm_size is None:
    wm_size = img.info.get('watermark_size', 9)
    redundancy = img.info.get('redundancy', 3)
    watermark_text = img.info.get('watermark_text', '')
    print(f"⚠️ Fallback: read metadata from .info dict (may not persist after save)")

print(f"\n📝 Stored metadata:")
print(f"   Size: {wm_size}×{wm_size}")
print(f"   Redundancy: {redundancy}")
print(f"   Text: '{watermark_text}'")

# Extract watermark
wm_rgb = np.array(img.convert('RGB'))
Y, _, _ = rgb_to_ycbcr(wm_rgb)
extracted = extract_watermark_dwt_dct(Y, wm_size, redundancy)

print(f"\n✅ Extraction Results:")
print(f"   Watermark shape: {extracted.shape}")
print(f"   Foreground density: {np.mean(extracted > 0):.3f}")
print(f"   Unique values: {np.unique(extracted)}")
print(f"\n📊 Pattern preview (first 3×3):")
print(extracted[:3, :3])
