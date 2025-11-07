# AI Image Processing Suite

A comprehensive Text Prompt to Watermarked Image generation web application that integrates three powerful modules:

1. **DF-GAN Text-to-Image Generation (Module 1)** - Generate stunning images from text descriptions
2. **Image Prompt Evaluator (Module 2)** - Evaluate how well AI-generated images match their text prompts
3. **Watermark Embedding (Both Visible and Invisible) (Module 3)** - Add professional watermarks to protect your images
4. **Website Integration (Full Suite)** - A cohesive web app that orchestrates DF-GAN generation (Module 1), CLIP-based evaluation (Module 2), and visible and invisible watermarking mechanisms (Module 3), adding batch workflows, auto-evaluation, galleries, recommendations, and streamlined downloads.

## Features

### 🎨 Text-to-Image Generation
- Generate images from text descriptions using the DF-GAN model
- Support for multiple datasets (CUB, COCO)
- Customizable generation settings (batch size, steps, guidance scale, seed)
- Example prompts for different datasets
- Real-time generation progress tracking

### 🔍 Image Quality Evaluation
- CLIP-based semantic similarity evaluation
- Keyword analysis and confidence scoring
- Adjustable similarity thresholds
- Quality classification (Excellent, Good, Fair, Poor)
- Detailed feedback and suggestions

### 🛡️ Watermark Protection

#### Visible Watermarking
- Add visible watermarks (image or text) to your images in bulk.
- **Features:**
  - Customizable watermark positioning: Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center.
  - Adjustable opacity (0–100%), scale (auto-resize), and rotation.
  - Padding options (pixels or percentage) for precise placement.
  - Batch processing with file naming options (prefix/suffix).
  - Real-time preview and batch download functionality.

#### Invisible Watermarking (DWT-DCT)
- Embed robust, imperceptible watermarks using a hybrid **Discrete Wavelet Transform + Discrete Cosine Transform (DWT-DCT)** algorithm.
- **Features:**
  - Watermark can be either text or image.
  - Watermark is binarized and adaptively resized to match the host image's capacity.
  - High redundancy embedding for robustness against attacks.
  - All metrics are computed and displayed in the web UI for transparency.

### � Authentication & User Management
- **Firebase Authentication** - Secure, cross-device authentication
- **Multiple Sign-in Methods:**
  - Google Sign-in (OAuth)
  - Email/Password authentication
- **User Features:**
  - Personal galleries for generated, evaluated, and watermarked images
  - Session management with 30-minute timeout
  - Secure JWT tokens for API authentication
  - Cross-device synchronization
- **Security:**
  - Firebase-managed password encryption
  - Token-based API authorization
  - Automatic session timeout on inactivity

### �📱 Modern UI/UX
- Responsive design for all device sizes
- Drag-and-drop file uploads
- Real-time progress indicators
- Real time Toast notification integration for flow indication
- Image gallery with filtering and search

## Technology Stack

- **Frontend**: React 18 with functional components and hooks
- **Backend**: Flask (Python) with SQLAlchemy ORM
- **Authentication**: Firebase Authentication (Google OAuth + Email/Password)
- **Database**: SQLite for user data and image records
- **Styling**: Tailwind CSS with custom components
- **Icons**: Heroicons
- **Routing**: React Router DOM
- **File Handling**: React Dropzone
- **Notifications**: React Hot Toast
- **Build Tool**: Create React App
- **AI Models**: 
  - DF-GAN for text-to-image generation
  - CLIP for image-prompt evaluation
  - DWT-DCT for invisible watermarking

## Server Setup and Path Configuration

The AI-Image-Suite server uses a flexible path configuration system that works across different development environments and systems.

### Quick Setup

1. **Clone the repositories**
   ```bash
   # Clone the main project
   git clone AI-Image-Suite
   cd AI-Image-Suite
   

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install dependencies**
   ```bash
   npm install
   ```

4. **Configure Firebase Authentication**
   
   See `QUICK_SETUP.md` for detailed Firebase setup instructions.
   
   Quick steps:
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Select your project
   - Enable **Email/Password** authentication in Authentication > Sign-in method
   - Configure `.env` file with your Firebase credentials:
     ```
     FIREBASE_PROJECT_ID=your-project-id
     ```

5. **Start the backend server**
   ```bash
   python server/app.py
   ```

6. **Start the development server**
   ```bash
   npm start
   ```

7. **Open your browser**
   Navigate to `http://localhost:3000`


## Modules at a Glance (Standalone References)
- Module 1: DF-GAN Text-to-Image Generation
  - GitHub: https://github.com/The-GANners/DF-GAN
  - Purpose: Generate images from text using GANs (CUB/COCO, etc.)
- Module 2: CLIP-based Image Prompt Evaluator
  - Github: https://github.com/The-GANners/Module-2
  - Purpose: Score semantic match between an image and prompt + keyword/feature analysis
- Module 3: Watermark UI (reference implementation)
  - Github: https://github.com/The-GANners/Watermark-UI
  - Purpose: Add visible or invisible (text or image) watermarks to images for protection and copyright verification.

## How Each Module Works

### Module 1 — DF-GAN Text-to-Image (Standalone)
- What it is:
  - A text-to-image GAN that synthesizes images from natural language prompts. 

- Inputs:
  - Prompt(s): tokenized via DAMSM vocabulary.
  - Noise Vector: z_dim (default 100), batch size, truncation/trunc_rate, manual_seed, encoder_epoch; dataset-specific text/image encoders.

- Outputs:
  - Generated PNGs (normalized from [-1,1] to [0,255]) saved per-sentence. Training also writes periodic grids and checkpoints to saved_models folder.
  - Metrics/artifacts: FID computed with 2048-dim Inception features against npz stats.
  - optional CLIP alignment grid and score saved under alignment_samples.

- Key components:
  - Text Encoder : bi-LSTM that returns word- and sentence-level embeddings (256).

  - Image encoder : (CNN_ENCODER): Inception v3 backbone projecting image features to 256-d, used by DAMSM and dataset prep.

  - Generator:
    - noise z → fc to 8·nf·4×4 → stack of G_Block(up-sample) layers (get_G_in_out_chs) → to_rgb → Tanh.
    - Text conditioning via DFBLK + Affine: concatenates [z, sent_emb] and modulates feature maps in each block.

  - Discriminator:
    - NetD extracts multi-scale features with D_Block downsamplers.
    - NetC concatenates image features with sentence embedding (spatially replicated) to produce a conditional real/fake logit.
    - Losses use hinge; includes mismatched text negatives.
   - Regularization: Matching-Aware Gradient Penalty on image/text gradients.
   - Stabilization: Exponential Moving Average of G, EMA weights are used for testing/FID.

  - Evaluation:
    - FID: InceptionV3 2048-d features vs dataset npz.
    - CLIP alignment (optional diagnostic): generates a grid and cosine scores for prompts using ViT-B/32 CLIP pre-trained model.

- Standalone repo: https://github.com/The-GANners/DF-GAN

### Module 2 — CLIP-based Image Prompt Evaluator (Standalone)
- What it is:
  - A Python module that evaluates image–prompt alignment using CLIP ViT-B/32, with enhanced keyword analysis, contradiction checks, and score normalization.

- Inputs:
  - image_path: path to a PNG/JPEG/WebP image.
  - prompt: natural-language text.
  
- Outputs (dict):
  - overall_score, original_score, percentage_match, original_percentage, raw_percentage
  - quality (Excellent/Good/Fair/Poor), feedback, prompt
  - keyword_analysis: [{ keyword, present, confidence, raw_score, status, status_type }]
  - meets_threshold: bool
  - detailed_metrics: { raw_score, penalized_score, average_score, normalized_score, percentage, all_scores }
  - contradiction_warning (if penalty applied)
  - missing_feature_analysis: { present_features|weak_features|missing_features (with importance_weight, confidence, raw_score), counts }
  - missing_feature_feedback, missing_feature_penalty, penalty_percentage

- How it works (from code):
  - CLIP ViT-B/32 encodes image and multiple prompt variants (“a photo of …”, “an image showing …”, cleaned prompt); uses best similarity.
  - Applies semantic contradiction penalties for conflicting concepts if contradiction score exceeds the main score by a margin.
  - Extracts important keywords via NLTK (POS tagging, stopwords), boosts animals/living things using WordNet, and assigns importance weights.
  - Probes each keyword with multiple templates to classify present/weak/missing.
  - Computes a weighted missing-feature penalty; combines with contradiction penalty and recalculates normalized percentage.
  - Normalizes raw CLIP scores to 0–100% with calibrated bands and maps to quality:
    - Excellent > 0.28, Good > 0.22, Fair > 0.18, else Poor.

- Standalone repo: https://github.com/The-GANners/Module-2
- Core CLIP model: https://github.com/openai/CLIP

### Module 3 — Watermark UI (Standalone)

**Visible Watermarking:**
- What it is:
  - A standalone Tkinter desktop application for batch visible watermarking of images.
  - Provides a dark-themed UI, drag-and-drop support, progress tracking, and safe overwrite behavior.
- Features:
  - Add image or text watermarks to multiple images at once.
  - Customizable position (Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center), opacity, scale (auto-resize), rotation, and padding (px or %).
  - Real-time preview and batch output with file renaming options.
- Inputs:
  - Images: selected via folder picker or multi-file selection.
  - Watermark: image or text, with options for font, color, and size.
  - Options: position, opacity, scale, rotation, padding, output naming.
- Outputs:
  - Watermarked images saved to the chosen output directory, with optional prefix/suffix renaming.

**Command to run the app :**
python FreeMark.py

**Invisible Watermarking (DWT-DCT):**

  - Embeds a binary watermark (text or image) into the host image using a combination of Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT).
  - The watermark is binarized and adapatively resized to fit the host image's embedding capacity.
  - Embedding is performed in the DCT coefficients of the DWT-LL subband, with redundancy for robustness.
  - Extraction is possible even after common image attacks (JPEG compression, noise, blur).
  - **Metrics & Robustness Testing:**
  - **Imperceptibility (PSNR):**  
    - The system computes the Peak Signal-to-Noise Ratio (PSNR) between the original and watermarked image.
    - High PSNR (e.g., >40 dB) means the watermark is visually imperceptible.
    - Formula:  
      `PSNR = 20 * log10(255.0 / sqrt(MSE))`  
      where MSE is the mean squared error between original and watermarked images.
  - **Robustness (NCC):**  
    - Normalized Cross-Correlation (NCC) is computed between the original and extracted watermark.
    - NCC close to 1.0 means the watermark is perfectly recovered.
    - Formula:  
      `NCC = mean(sign(original_wm) * sign(recovered_wm))`
  - **Attack Simulation:**  
    - Robustness is tested against common image attacks:
      - JPEG compression (Q=85, 70, 50)
      - Gaussian noise (σ=0.03, 0.06)
      - Gaussian blur (σ=0.8, 1.2)
    - The system reports PSNR and NCC for each attack, showing how well the watermark survives.

- **Implementation:**
  - Embedding and extraction are performed on the Y channel (luminance) in YCbCr color space.
  - Watermark is embedded in DCT coefficients of DWT-LL subbands, with adaptive margin and redundancy.
  
- Outputs:
  - Watermarked images and extracted watermark images saved to disk.
  - Computed PSNR/NCC metrics for verification.

  - **Command to embed watermark:**
  python watermarkdwt.py

  - **Command to verify watermark:**
  For text watermark:
  python verify_watermark.py -i "Watermarked-image-path" -t "Text-which-was-watermarked"

  For image watermark:
  python verify_watermark.py -o "Watermarked-image-path" -t "Path-of-image-which-was-used-as-watermark"

- Standalone repo: https://github.com/The-GANners/Watermark-UI

