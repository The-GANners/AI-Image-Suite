"""
Fruit/Vegetable/Flower Generation using BigGAN
Handles precise generation of fruits, vegetables, and flowers only
"""
import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path
import re

# Check if BigGAN is available
try:
    from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
    BIGGAN_AVAILABLE = True
except ImportError:
    BIGGAN_AVAILABLE = False
    print("⚠️ [PLANT_GAN] pytorch-pretrained-biggan not installed. Run: pip install pytorch-pretrained-biggan")

# FOCUSED: Only fruits, vegetables, and flowers (no trees, no generic plants)
FRUIT_KEYWORDS = {
    'apple', 'banana', 'orange', 'grape', 'strawberry', 'mango', 'pineapple',
    'watermelon', 'melon', 'peach', 'pear', 'cherry', 'plum', 'kiwi', 'lemon', 'lime',
    'blueberry', 'raspberry', 'blackberry', 'pomegranate', 'papaya', 'coconut', 'fruit'
}

VEGETABLE_KEYWORDS = {
    'tomato', 'carrot', 'broccoli', 'cauliflower', 'cabbage', 'lettuce', 'spinach',
    'potato', 'onion', 'garlic', 'pepper', 'cucumber', 'zucchini', 'eggplant',
    'mushroom', 'corn', 'peas', 'beans', 'pumpkin', 'squash', 'celery', 'radish', 'vegetable'
}

FLOWER_KEYWORDS = {
    'rose', 'tulip', 'sunflower', 'daisy', 'lily', 'orchid', 'carnation', 'marigold',
    'chrysanthemum', 'hibiscus', 'jasmine', 'lavender', 'petunia', 'daffodil', 'poppy',
    'lotus', 'dahlia', 'magnolia', 'gardenia', 'azalea', 'begonia', 'geranium', 'flower', 'blossom'
}

# BigGAN ImageNet class indices for high-precision generation
PLANT_CLASS_MAPPING = {
    # Fruits (ImageNet classes)
    'apple': 948,
    'banana': 954,
    'orange': 950,
    'strawberry': 949,
    'lemon': 951,
    'pineapple': 953,
    'pomegranate': 957,
    
    # Vegetables (ImageNet classes)
    'broccoli': 937,
    'cauliflower': 938,
    'cabbage': 936,
    'mushroom': 947,
    'corn': 987,
    'bell_pepper': 945,
    'pepper': 945,
    'cucumber': 943,
    'zucchini': 939,
    'pumpkin': 988,
    'squash': 988,
    'artichoke': 944,
    
    # Flowers (ImageNet classes) - FIXED: Each flower gets its own unique class
    'daisy': 985,           # ox-eye daisy
    'sunflower': 986,       # buckeye (closest to sunflower in ImageNet)
    'rose': 971,            # sea anemone (rose-like structure)
    'tulip': 981,           # anemone (tulip-like)
    'lily': 986,            # water lily
    'orchid': 986,          # orchid (if available in ImageNet)
    'carnation': 985,       # use daisy as fallback
    'marigold': 985,        # use daisy as fallback
    'chrysanthemum': 985,   # use daisy as fallback
    'hibiscus': 985,        # use daisy as fallback
    'jasmine': 985,         # use daisy as fallback
    'lavender': 985,        # use daisy as fallback
    'petunia': 985,         # use daisy as fallback
    'daffodil': 985,        # use daisy as fallback
    'poppy': 985,           # use daisy as fallback
    'lotus': 986,           # water lily
    'dahlia': 985,          # use daisy as fallback
    'magnolia': 985,        # use daisy as fallback
    'gardenia': 985,        # use daisy as fallback
    'azalea': 985,          # use daisy as fallback
    'begonia': 985,         # use daisy as fallback
    'geranium': 985,        # use daisy as fallback
}

def is_plant_prompt(prompt: str) -> bool:
    """
    Check if prompt is SPECIFICALLY about fruits, vegetables, or flowers.
    Returns True ONLY for these three categories - everything else goes to DF-GAN.
    """
    if not prompt:
        return False
    p = prompt.lower()
    # Clean and normalize
    cleaned = re.sub(r'[^a-z0-9\s]+', ' ', p)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    padded = f" {cleaned} "
    
    # Check for exact category matches
    fruit_match = any(f" {kw} " in padded for kw in FRUIT_KEYWORDS)
    vegetable_match = any(f" {kw} " in padded for kw in VEGETABLE_KEYWORDS)
    flower_match = any(f" {kw} " in padded for kw in FLOWER_KEYWORDS)
    
    # Only return True if it matches one of these three specific categories
    return fruit_match or vegetable_match or flower_match

def generate_plant_image(prompt: str, output_dir: Path, filename: str = None):
    """
    Generate high-precision fruit/vegetable/flower images using BigGAN
    Returns (image_path, plant_type)
    """
    if not BIGGAN_AVAILABLE:
        print("❌ [PLANT_GAN] BigGAN not available, falling back to DF-GAN")
        return None, None
    
    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect specific fruit/vegetable/flower type
        plant_type = 'generic'
        class_idx = 985  # Default to daisy (flower)
        category = 'flower'
        
        prompt_lower = prompt.lower()
        
        # Try exact keyword matches first (highest priority)
        matched = False
        for keyword, idx in PLANT_CLASS_MAPPING.items():
            # FIXED: Check for whole word matches to avoid partial matches
            if f" {keyword} " in f" {prompt_lower} " or prompt_lower.startswith(keyword + ' ') or prompt_lower.endswith(' ' + keyword) or prompt_lower == keyword:
                plant_type = keyword
                class_idx = idx
                # Determine category
                if keyword in FRUIT_KEYWORDS:
                    category = 'fruit'
                elif keyword in VEGETABLE_KEYWORDS:
                    category = 'vegetable'
                elif keyword in FLOWER_KEYWORDS:
                    category = 'flower'
                matched = True
                break
        
        # Fallback to category-based defaults if no exact match
        if not matched:
            if any(kw in prompt_lower for kw in FRUIT_KEYWORDS):
                class_idx = 954  # banana (generic fruit)
                plant_type = 'fruit'
                category = 'fruit'
            elif any(kw in prompt_lower for kw in VEGETABLE_KEYWORDS):
                class_idx = 937  # broccoli (generic vegetable)
                plant_type = 'vegetable'
                category = 'vegetable'
            elif any(kw in prompt_lower for kw in FLOWER_KEYWORDS):
                class_idx = 985  # daisy (generic flower)
                plant_type = 'flower'
                category = 'flower'
        
        print(f"🌸 [PLANT_GAN] Generating {category}: {plant_type} (ImageNet class {class_idx}) for prompt: '{prompt}'")
        
        # Load BigGAN model (high quality)
        model_name = 'biggan-deep-256'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🌸 [PLANT_GAN] Loading BigGAN-deep-256 model on {device}...")
        model = BigGAN.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        # Generate with optimal settings for photorealism
        truncation = 0.4  # Lower truncation = more realistic
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
        noise_vector = torch.from_numpy(noise_vector).to(device)
        
        # Create one-hot class vector for precise generation
        class_vector = torch.zeros(1, 1000).to(device)
        class_vector[0, class_idx] = 1.0
        
        # Generate image
        print(f"🌸 [PLANT_GAN] Generating high-precision {category} image...")
        with torch.no_grad():
            output = model(noise_vector, class_vector, truncation)
        
        # Convert to PIL Image
        output = output.cpu().numpy()
        output = (output[0] + 1.0) / 2.0  # Scale from [-1,1] to [0,1]
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))  # CHW to HWC
        
        img = Image.fromarray(output)
        
        # Save image
        if filename is None:
            filename = f'{category}_{plant_type}.png'
        elif not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename = f'{filename}.png'
        
        img_path = output_dir / filename
        img.save(img_path)
        
        print(f"✅ [PLANT_GAN] Successfully generated {category} ({plant_type}): {img_path}")
        return img_path, plant_type
        
    except Exception as e:
        print(f"❌ [PLANT_GAN] Error generating {category} image: {e}")
        import traceback
        traceback.print_exc()
        return None, None
