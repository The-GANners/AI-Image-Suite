## 🎨 AI-Image-Suite: Text-to-Image AI Pipeline with Semantic Evaluation and Visible & DWT-DCT Invisible Watermarking



<div align="center">
  
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-EE4C2C.svg?logo=pytorch)
![CLIP](https://img.shields.io/badge/OpenAI%20CLIP-Model-412991.svg?logo=openai)
![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB.svg?logo=react)
![Flask](https://img.shields.io/badge/Backend-Flask-000000.svg?logo=flask)
![Firebase](https://img.shields.io/badge/Auth-Firebase-FFCA28.svg?logo=firebase)

</div>

---

## 📖 Overview

### *A comprehensive Text Prompt to Watermarked Image generation web application that integrates three powerful modules* ✨

1.  **DF-GAN Text-to-Image Generation (Module 1)** - Generate stunning images from text descriptions
2.  **Image Prompt Evaluator (Module 2)** - Evaluate how well AI-generated images match their text prompts
3.  **Watermark Embedding (Both Visible and Invisible) (Module 3)** - Add professional watermarks to protect your images
4.  **Website Integration (Full Suite)** - A cohesive web app that orchestrates DF-GAN generation (Module 1), CLIP-based evaluation (Module 2), and visible and invisible watermarking mechanisms (Module 3), adding batch workflows, auto-evaluation, galleries, recommendations, and streamlined downloads.

> 💡 **Key Features:** Full-stack architecture, DF-GAN for text to image generation, secure authentication, Semantic Evaluation through CLIP and DWT-DCT invisible watermarking for high robustness.

---

## 🏗️ Features

### 🎨 Text-to-Image Generation
* Generate images from text descriptions using the **DF-GAN model**
* Support for multiple datasets (**CUB, COCO**)
* Customizable generation settings (batch size, steps, guidance scale, seed)
* Example prompts for different datasets
* Real-time generation progress tracking

### 🔍 Image Quality Evaluation
* **CLIP-based semantic similarity evaluation**
* Keyword analysis and confidence scoring
* Adjustable similarity thresholds
* Quality classification (**Excellent, Good, Fair, Poor**)
* Detailed feedback and suggestions

### 🛡️ Watermark Protection

#### Visible Watermarking
* Add visible watermarks (image or text) to your images in bulk.
* **Features:**
    * Customizable watermark positioning: Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center.
    * Adjustable opacity (0–100%), scale (auto-resize), and rotation.
    * Padding options (pixels or percentage) for precise placement.
    * Real-time preview and batch download functionality.

#### Invisible Watermarking (DWT-DCT)
* Embed robust, imperceptible watermarks using a hybrid **Discrete Wavelet Transform + Discrete Cosine Transform (DWT-DCT)** algorithm.
* **Features:**
    * Watermark can be either text or image.
    * Watermark is binarized and adaptively resized to match the host image's capacity.
    * High redundancy embedding for **robustness against attacks**.
    * All metrics are computed and displayed in the web UI for transparency.
    *  Real-time preview and batch download functionality.
    *  Support for Watermark Extraction (Binary Pattern) with verification functionality.

### 🔐 Authentication & User Management
* **Firebase Authentication** - Secure, cross-device authentication
* **Multiple Sign-in Methods:**
    * Google Sign-in (OAuth)
    * Email/Password authentication
* **User Features:**
    * Personal galleries for generated, evaluated, and watermarked images
    * Session management with 30-minute timeout
    * Secure **JWT tokens** for API authentication
    * Cross-device synchronization
* **Security:**
    * Firebase-managed password encryption
    * Token-based API authorization
    * Automatic session timeout on inactivity

### 📱 Modern UI/UX
* Responsive design for all device sizes
* Drag-and-drop file uploads
* Real-time progress indicators
* Real time Toast notification integration for flow indication
* Image gallery with filtering and search

---

## 💻 Technology Stack

| Category | Components |
| :--- | :--- |
| **Frontend** | React 18 with functional components and hooks |
| **Backend** | Flask (Python) with SQLAlchemy ORM |
| **Authentication** | Firebase Authentication (Google OAuth + Email/Password) |
| **Database** | SQLite for user data and image records |
| **Styling** | Tailwind CSS with custom components, Heroicons |
| **Routing** | React Router DOM |
| **File Handling** | React Dropzone |
| **Notifications** | React Hot Toast |
| **Build Tool** | Create React App |
| **AI Models** | **DF-GAN** for text-to-image generation, **CLIP** for image-prompt evaluation |
| **Watermarking mechanism** | Simple **visible** watermarking, **DWT-DCT** for invisible watermarking |

---

## 🚀 Server Setup and Path Configuration

The AI-Image-Suite server uses a flexible path configuration system that works across different development environments and systems.

### Quick Setup

1.  **Clone the repositories**
    ```bash
    # Clone the main project
    git clone AI-Image-Suite
    cd AI-Image-Suite
    ```

2.  **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install dependencies**
    ```bash
    npm install
    ```

4.  **Configure Firebase Authentication**
    
    See `QUICK_SETUP.md` for detailed Firebase setup instructions.
    
    Quick steps:
    - Go to [Firebase Console](https://console.firebase.google.com/)
    - Select your project
    - Enable **Email/Password** authentication in Authentication > Sign-in method
    - Configure `.env` file with your Firebase credentials:
    ```
    FIREBASE_PROJECT_ID=your-project-id
    ```

5. **Start the backend and development server simultaneously:**

   ⚠️ **Important Note:**  
   When starting both servers, the frontend will open immediately, but the **initialization of the database, models, and encoders** may take **10–15 seconds** on the first run. After this warm-up, the application responds normally and runs quickly.

   ```bash
   npm run dev


6.  **Open your browser**
    Navigate to `http://localhost:3000`

---

## 🧠 Modules at a Glance (Standalone References)

| Module | Purpose | GitHub Repository |
| :--- | :--- | :--- |
| **Module 1: DF-GAN** | Generate images from text using GANs (CUB/COCO, etc.) | [https://github.com/The-GANners/DF-GAN](https://github.com/The-GANners/DF-GAN) |
| **Module 2: CLIP Evaluator** | Score semantic match between image & prompt | [https://github.com/The-GANners/Module-2](https://github.com/The-GANners/Module-2) |
| **Module 3: Watermark UI** | Add visible or invisible watermarks | [https://github.com/The-GANners/Watermark-UI](https://github.com/The-GANners/Watermark-UI) |


---

## ⚙️ How Each Module Works

### Module 1 — DF-GAN Text-to-Image (Standalone)

* **What it is:** A text-to-image **GAN** that synthesizes images from natural language prompts.
* **Key components:** 
    * **Text Encoder:** bi-LSTM that returns word- and sentence-level embeddings (256).
   * **Image encoder:** (`CNN_ENCODER`): Inception v3 backbone projecting image features to 256-d, used by DAMSM and dataset prep.
  * **Generator:** Text conditioning via **DFBLK + Affine**: concatenates $[z, \text{`sent_emb`}]$ and modulates feature maps in each block.
    * **Discriminator:** **NetD** extracts multi-scale features; **NetC** concatenates image features with sentence embedding to produce a conditional real/fake logit.
* **Regularization:** Matching-Aware Gradient Penalty on image/text gradients.
* **Stabilization:** Exponential Moving Average of G, EMA weights are used for testing/FID.

### Module 2 — CLIP-based Image Prompt Evaluator (Standalone)

* **What it is:** A Python module that evaluates image–prompt alignment using **CLIP ViT-B/32**, with enhanced keyword analysis, contradiction checks, and score normalization.
* **How it works (from code):** 
    * **CLIP ViT-B/32** encodes image and multiple prompt variants; uses best similarity.
    * Applies **semantic contradiction penalties** for conflicting concepts.
    * Extracts important keywords via NLTK, boosts animals/living things using WordNet, and assigns importance weights.
    * Computes a **weighted missing-feature penalty**; combines with contradiction penalty and recalculates normalized percentage.
    * Normalizes raw CLIP scores to $0–100\%$ with calibrated bands and maps to quality: Excellent $> 0.28$, Good $> 0.22$, Fair $> 0.18$, else Poor. 

### Module 3 — Watermark UI (Standalone)

### Visible Watermarking
* **Description:**  Add visible watermarks (image or text) to your images in bulk.
* **Implementation:** Customizable watermark positioning with adjustable opacity (0–100%) and support for various padding options (pixels or percentages).
  
#### Invisible Watermarking (DWT-DCT)
* **Description:** Embeds a binary watermark (text or image) into the host image using a combination of **Discrete Wavelet Transform (DWT)** and **Discrete Cosine Transform (DCT)**.
* **Implementation:**  Embedding and extraction are performed on the $\text{Y}$ channel (luminance) in $\text{YCbCr}$ color space. Watermark is embedded in **DCT coefficients of DWT-LL subbands**, with adaptive margin and redundancy. Supports watermark extraction (binary pattern) with verification functionality.

| Metric | Purpose | Formula |
| :--- | :--- | :--- |
| **Imperceptibility (PSNR)** | High PSNR (e.g., >40 dB) means the watermark is visually imperceptible. | $$PSNR = 20 \cdot \log_{10} \left(\frac{255.0}{\sqrt{MSE}}\right)$$ |
| **Robustness (NCC)** | NCC close to 1.0 means the watermark is perfectly recovered. | $$NCC = mean(sign(original_wm) \cdot sign(recovered_wm))$$ |
| **Attack Simulation** | Robustness is tested against common image attacks: JPEG compression ($Q=85, 70, 50$), Gaussian noise ($\sigma=0.03, 0.06$), Gaussian blur ($\sigma=0.8, 1.2$). |

---

## 📄 License

This project is released under the **MIT License**.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions and feedback:
* Open an issue on GitHub
* Join our community discussions

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star! ⭐

**Made with ❤️ by the GANners Team**

</div>
