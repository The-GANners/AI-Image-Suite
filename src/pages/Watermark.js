import React, { useState } from 'react';
import ExtractWatermarkPopup from '../components/ExtractWatermarkPopup';
import { useDropzone } from 'react-dropzone';
import { 
  PencilSquareIcon, 
  CloudArrowUpIcon,
  AdjustmentsHorizontalIcon,
  DocumentArrowDownIcon,
  EyeIcon,
  PhotoIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import { useAuth } from '../contexts/AuthContext';

const Watermark = () => {
  const { requireAuth } = useAuth();
  const [images, setImages] = useState([]);
  const [watermarkImage, setWatermarkImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedImages, setProcessedImages] = useState([]);
  const [previewGrid, setPreviewGrid] = useState([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [settings, setSettings] = useState({
    position: 'SE',
    opacity: 80, // percent (0-100)
    scale: 20,
    paddingX: 5,
    paddingY: 5,
    paddingUnit: 'percentage',
    autoResize: true,
    rotation: 0 // rotation angle in degrees (0-360)
  });

  const [watermarkType, setWatermarkType] = useState('visible'); // 'visible' | 'invisible'
  const [watermarkMode, setWatermarkMode] = useState('image'); // 'image' | 'text'
  const [watermarkFile, setWatermarkFile] = useState(null);
  const [watermarkText, setWatermarkText] = useState('Sample Watermark');
  const [textSize, setTextSize] = useState(60);
  const [textColor, setTextColor] = useState('#FFFFFF');
  const [invisibleAlpha, setInvisibleAlpha] = useState(0.28); // Embedding strength for invisible watermark
  
  // Robustness testing states
  const [showRobustnessModal, setShowRobustnessModal] = useState(false);
  const [robustnessResults, setRobustnessResults] = useState(null);
  const [isTestingRobustness, setIsTestingRobustness] = useState(false);

  // Extract Watermark popup state
  const [showExtractPopup, setShowExtractPopup] = useState(false);

  const positions = [
    { value: 'top-left', label: 'Top Left' },
    { value: 'top-right', label: 'Top Right' },
    { value: 'bottom-left', label: 'Bottom Left' },
    { value: 'bottom-right', label: 'Bottom Right' },
    { value: 'center', label: 'Center' }
  ];

  const { getRootProps: getImageRootProps, getInputProps: getImageInputProps } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    onDrop: (acceptedFiles) => {
      const newImages = acceptedFiles.map(file => ({
        id: Date.now() + Math.random(),
        file,
        preview: URL.createObjectURL(file),
        name: file.name
      }));
      setImages(prev => [...prev, ...newImages]);
    },
    multiple: true
  });

  const { getRootProps: getWatermarkRootProps, getInputProps: getWatermarkInputProps } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setWatermarkImage({
          file,
          preview: URL.createObjectURL(file),
          name: file.name
        });
      }
    },
    multiple: false
  });

  const removeImage = (id) => {
    setImages(prev => prev.filter(img => img.id !== id));
  };

  const fileToDataUrl = (file) => new Promise((resolve, reject) => {
    try {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    } catch (e) { reject(e); }
  });
  const urlToDataUrl = async (url) => {
    // if it's already a data URL
    if (typeof url === 'string' && url.startsWith('data:')) return url;
    const res = await fetch(url);
    const blob = await res.blob();
    return await fileToDataUrl(blob);
  };
  const imageItemToDataUrl = async (item) => {
    if (item?.file) return await fileToDataUrl(item.file);
    if (item?.preview) return await urlToDataUrl(item.preview);
    throw new Error('Invalid image item');
  };
  const getWatermarkDataUrl = async () => {
    const f = watermarkFile || watermarkImage?.file;
    if (!f) return null;
    return await fileToDataUrl(f);
  };
  const mapPos = (p) => {
    // Accept both human labels and codes, normalize to NW/NE/SW/SE
    const v = (p || '').toUpperCase();
    if (['NW','NE','SW','SE'].includes(v)) return v;
    const map = {
      'TOP-LEFT':'NW', 'TOP RIGHT':'NE', 'TOP-RIGHT':'NE',
      'BOTTOM-LEFT':'SW', 'BOTTOM RIGHT':'SE', 'BOTTOM-RIGHT':'SE', 'CENTER':'SE'
    };
    return map[v] || 'SE';
  };
  const mapUnit = (u) => (u === '%' || (u || '').toLowerCase().startsWith('per')) ? '%' : 'px';

  const handleProcessClick = () => {
    requireAuth(handleProcess);
  };

  const handleProcess = async () => {
    if (images.length === 0) {
      toast.error('Please select images to watermark');
      return;
    }
    if (watermarkMode === 'image' && !(watermarkFile || watermarkImage?.file)) {
      toast.error('Please select a watermark image');
      return;
    }
    setIsProcessing(true);
    try {
      // Get token for authenticated requests (optional - saves to gallery if logged in)
      const token = localStorage.getItem('token');
      const headers = {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` })
      };

      let payload, endpoint;
      const imgsPayload = await Promise.all(images.map(async (img, idx) => ({
        name: img.name || `image_${idx + 1}.png`,
        url: await imageItemToDataUrl(img)
      })));

      if (watermarkType === 'invisible') {
        // Invisible watermark payload
        endpoint = '/api/watermark/apply-invisible';
        payload = {
          images: imgsPayload,
          watermarkMode: watermarkMode, // 'text' or 'image'
          alpha: invisibleAlpha
        };
        
        if (watermarkMode === 'image') {
          const wmDataUrl = await getWatermarkDataUrl();
          if (!wmDataUrl) throw new Error('Watermark image missing');
          payload.watermarkDataUrl = wmDataUrl;
        } else {
          payload.watermarkText = (watermarkText || '').trim();
        }
      } else {
        // Visible watermark payload (existing code)
        endpoint = '/api/watermark/apply';
        const unit = mapUnit(settings.paddingUnit);
        const opacityFactor = Math.max(0, Math.min(1, (settings.opacity ?? 100) / 100));
        payload = {
          images: imgsPayload,
          mode: watermarkMode === 'text' ? 'text' : 'image',
          pos: mapPos(settings.position),
          padding: { x: parseInt(settings.paddingX || 0, 10), xUnit: unit, y: parseInt(settings.paddingY || 0, 10), yUnit: unit },
          scale: !!settings.autoResize,
          opacity: opacityFactor,
          rotation: parseInt(settings.rotation || 0, 10)
        };
        // NEW: add top-level padding aliases for backend normalizer
        payload.padx = parseInt(settings.paddingX || 0, 10);
        payload.pady = parseInt(settings.paddingY || 0, 10);
        payload.xUnit = unit;
        payload.yUnit = unit;
        if (payload.mode === 'image') {
          const wmDataUrl = await getWatermarkDataUrl();
          if (!wmDataUrl) throw new Error('Watermark image missing');
          payload.watermarkDataUrl = wmDataUrl;
        } else {
          payload.text = (watermarkText || '').trim();
          payload.textSize = parseInt(textSize || 32, 10);
          payload.textColor = textColor || '#FFFFFF';
        }
      }
      
      console.log(`🚀 [WATERMARK] Sending ${watermarkType} watermark payload to backend:`, {
        type: watermarkType,
        mode: watermarkMode,
        endpoint
      });
      
      // Call backend
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      // Map response to UI list
      const processed = (data.images || []).map((out, i) => {
        const original = images[i];
        const base = (original?.name || `image_${i + 1}`).split('.');
        const ext = base.length > 1 ? base.pop() : 'png';
        const baseName = base.join('.');
        const processedName = `${baseName}.${ext}`;
        return {
          id: original?.id || Date.now() + Math.random(),
          originalName: original?.name || `image_${i + 1}.${ext}`,
          processedName,
          preview: out.dataUrl,
          downloadUrl: out.dataUrl,
          settings: { ...settings, mode: payload.mode }
        };
      });
      setProcessedImages(processed);
      toast.success(`Successfully watermarked ${processed.length} image(s)!`);
      
      // Auto-save to gallery if user is authenticated
      const authToken = localStorage.getItem('ai_image_suite_auth_token');
      if (authToken) {
        // Save all watermarked images to gallery
        processed.forEach(async (image) => {
          try {
            const base64Data = image.downloadUrl.split(',')[1];
            const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5001';
            await fetch(`${API_BASE}/api/gallery/save-watermarked`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
              },
              body: JSON.stringify({
                imageData: base64Data,
                originalName: image.originalName,
                watermarkText: watermarkText,
                watermarkPosition: settings.position,
                watermarkOpacity: settings.opacity
              })
            });
          } catch (error) {
            console.error('Failed to auto-save watermarked image to gallery:', error);
          }
        });
      }
    } catch (err) {
      console.error('Watermarking failed:', err);
      toast.error(`Failed: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadAll = () => {
    processedImages.forEach(image => {
      const link = document.createElement('a');
      link.href = image.downloadUrl;
      // FORCE PNG extension to prevent browser JPEG conversion
      const forcedName = image.processedName.replace(/\.(jpg|jpeg)$/i, '.png');
      link.download = forcedName;
      link.click();
    });
    toast.success(`Downloading ${processedImages.length} image(s) as PNG (lossless)`);
  };

  const handleTestRobustness = async () => {
    if (images.length === 0) {
      toast.error('Please select at least one image to test');
      return;
    }
    if (watermarkMode === 'image' && !(watermarkFile || watermarkImage?.file)) {
      toast.error('Please select a watermark image');
      return;
    }
    
    setIsTestingRobustness(true);
    try {
      // Get token for authenticated requests
      const token = localStorage.getItem('token');
      const headers = {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` })
      };

      // FIXED: Use ORIGINAL unwatermarked image for testing
      const firstImage = images[0];
      const imageDataUrl = await imageItemToDataUrl(firstImage);
      
      const payload = {
        images: [{
          name: firstImage.name || 'test.png',
          url: imageDataUrl  // FIXED: Changed from 'dataUrl' to 'url' to match backend expectation
        }],
        watermarkMode: watermarkMode,
        alpha: invisibleAlpha
      };
      
      if (watermarkMode === 'image') {
        const wmDataUrl = await getWatermarkDataUrl();
        if (!wmDataUrl) throw new Error('Watermark image missing');
        payload.watermarkDataUrl = wmDataUrl;
      } else {
        payload.watermarkText = (watermarkText || '').trim();
      }

      const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5001';
      console.log('🧪 [ROBUSTNESS TEST] Starting robustness test...', { API_BASE, alpha: invisibleAlpha });

      const res = await fetch(`${API_BASE}/api/watermark/test-robustness`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      console.log('✅ [ROBUSTNESS TEST] Test completed:', data);
      setRobustnessResults(data);
      setShowRobustnessModal(true);
      toast.success('Robustness test completed!');

    } catch (err) {
      console.error('Robustness testing failed:', err);
      toast.error(`Failed: ${err.message}`);
    } finally {
      setIsTestingRobustness(false);
    }
  };

  // NEW: import selected images from TextToImage via sessionStorage
  React.useEffect(() => {
    const raw = sessionStorage.getItem('watermarkSelectionData');
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        const selected = (parsed.images || []).map(item => ({
          id: Date.now() + Math.random(),
          file: null, // coming from generator, no File object
          preview: item.url,
          name: item.name || 'selected.png'
        }));
        if (selected.length > 0) {
          setImages(prev => [...selected, ...prev]);
        }
      } catch (e) {
        // ignore parse errors
      } finally {
        sessionStorage.removeItem('watermarkSelectionData');
      }
    }
  }, []);

  // NEW: build payload (shared shape with handleProcess)
  const buildPayload = async (useSubset = false) => {
    const srcImgs = useSubset ? images.slice(0, Math.min(6, images.length)) : images;
    const imgsPayload = await Promise.all(
      srcImgs.map(async (img, idx) => ({
        name: img.name || `image_${idx + 1}.png`,
        url: await imageItemToDataUrl(img)
      }))
    );

    let payload, endpoint;
    
    if (watermarkType === 'invisible') {
      // Invisible watermark payload
      endpoint = '/api/watermark/apply-invisible';
      payload = {
        images: imgsPayload,
        watermarkMode: watermarkMode, // 'text' or 'image'
        alpha: invisibleAlpha
      };
      
      if (watermarkMode === 'image') {
        const wmDataUrl = await getWatermarkDataUrl();
        if (!wmDataUrl) return null;
        payload.watermarkDataUrl = wmDataUrl;
      } else {
        payload.watermarkText = (watermarkText || '').trim();
      }
    } else {
      // Visible watermark payload
      endpoint = '/api/watermark/apply';
      const unit = mapUnit(settings.paddingUnit);
      const opacityFactor = Math.max(0, Math.min(1, (settings.opacity ?? 100) / 100));
      payload = {
        images: imgsPayload,
        mode: watermarkMode === 'text' ? 'text' : 'image',
        pos: mapPos(settings.position),
        padding: {
          x: parseInt(settings.paddingX || 0, 10),
          xUnit: unit,
          y: parseInt(settings.paddingY || 0, 10),
          yUnit: unit
        },
        scale: !!settings.autoResize,
        opacity: opacityFactor,
        rotation: parseInt(settings.rotation || 0, 10)
      };
      // NEW: add top-level padding aliases for backend normalizer
      payload.padx = parseInt(settings.paddingX || 0, 10);
      payload.pady = parseInt(settings.paddingY || 0, 10);
      payload.xUnit = unit;
      payload.yUnit = unit;
      if (payload.mode === 'image') {
        const wmDataUrl = await getWatermarkDataUrl();
        if (!wmDataUrl) return null; // caller handles
        payload.watermarkDataUrl = wmDataUrl;
      } else {
        payload.text = (watermarkText || '').trim();
        payload.textSize = parseInt(textSize || 32, 10);
        payload.textColor = textColor || '#FFFFFF';
      }
    }
    
    return { payload, endpoint };
  };

  // NEW: server-backed live preview (debounced, cancellable)
  React.useEffect(() => {
    let cancelled = false;
    let controller = new AbortController();
    const run = async () => {
      if (images.length === 0) {
        setPreviewGrid([]);
        return;
      }
      if (watermarkMode === 'image' && !(watermarkFile || watermarkImage?.file)) {
        setPreviewGrid([]);
        return;
      }
      // Debounce a bit to batch rapid changes
      await new Promise(r => setTimeout(r, 200));
      if (cancelled) return;

      setPreviewLoading(true);
      try {
        const result = await buildPayload(true);
        if (!result || !result.payload) {
          setPreviewGrid([]);
          setPreviewLoading(false);
          return;
        }
        
        const { payload, endpoint } = result;
        
        // Get token for authenticated requests (optional - saves to gallery if logged in)
        const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5001';
        const token = localStorage.getItem('token');
        const headers = {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        };
        
        console.log('🔍 [PREVIEW] Calling watermark preview API...', { API_BASE, endpoint, imageCount: payload.images.length });
        
        const res = await fetch(`${API_BASE}${endpoint}`, {
          method: 'POST',
          headers: headers,
          body: JSON.stringify(payload),
          signal: controller.signal
        });
        
        console.log('🔍 [PREVIEW] API response status:', res.status);
        
        const data = await res.json().catch(() => ({}));
        if (cancelled) return;
        if (!res.ok || data.error) {
          // On preview error, just clear previews silently
          console.warn('⚠️ [PREVIEW] API error:', data.error || `HTTP ${res.status}`);
          setPreviewGrid([]);
          setPreviewLoading(false);
          return;
        }
        const subset = images.slice(0, Math.min(6, images.length));
        const thumbs = (data.images || []).map((out, i) => ({
          id: subset[i]?.id || `preview-${i}`,
          name: subset[i]?.name || `image_${i + 1}.png`,
          url: out.dataUrl
        }));
        console.log('✅ [PREVIEW] Preview generated successfully:', thumbs.length, 'images');
        setPreviewGrid(thumbs);
      } catch (e) {
        if (!cancelled) {
          console.error('❌ [PREVIEW] Preview failed:', e);
          setPreviewGrid([]);
        }
      } finally {
        if (!cancelled) setPreviewLoading(false);
      }
    };
    run();

    return () => {
      cancelled = true;
      try { controller.abort(); } catch {}
    };
  }, [
    images,
    watermarkType,
    watermarkMode,
    watermarkFile,
    watermarkImage,
    watermarkText,
    textSize,
    textColor,
    invisibleAlpha,
    settings.position,
    settings.paddingX,
    settings.paddingY,
    settings.paddingUnit,
    settings.opacity,
    settings.autoResize,
    settings.rotation
  ]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Image Watermarking
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg max-w-2xl mx-auto">
            Protect your AI-generated images by adding professional watermarks. 
            Upload your images and watermark, then customize the settings.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Panel */}
          <div className="space-y-6">
            {/* Image Upload */}
            <div className="card p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <PhotoIcon className="w-5 h-5 text-primary-600" />
                <span>Select Images to Watermark</span>
              </h2>
              
              <div
                {...getImageRootProps()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-400 transition-colors cursor-pointer"
              >
                <input {...getImageInputProps()} />
                <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg text-gray-600 dark:text-gray-400 mb-2">
                  Drag & drop images here or click to select
                </p>
                <p className="text-sm text-gray-400 dark:text-gray-500">
                  Support for PNG, JPG, JPEG, WebP • Multiple files allowed
                </p>
              </div>

              {images.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                    Selected Images ({images.length})
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                    {images.map(image => (
                      <div key={image.id} className="relative group">
                        <img
                          src={image.preview}
                          alt={image.name}
                          className="w-full aspect-square object-cover rounded-lg"
                        />
                        <button
                          onClick={() => removeImage(image.id)}
                          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs hover:bg-red-600 opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          ×
                        </button>
                        <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-2 rounded-b-lg opacity-0 group-hover:opacity-100 transition-opacity">
                          {image.name}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Watermark Upload */}
            <div className="card p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <PencilSquareIcon className="w-5 h-5 text-primary-600" />
                <span>Select Watermark Image</span>
              </h2>
              
              <div
                {...getWatermarkRootProps()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-400 transition-colors cursor-pointer"
              >
                <input {...getWatermarkInputProps()} />
                {watermarkImage ? (
                  <div className="space-y-4">
                    <img 
                      src={watermarkImage.preview} 
                      alt="Watermark" 
                      className="max-h-32 mx-auto rounded-lg"
                    />
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {watermarkImage.name} • Click to replace
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <PencilSquareIcon className="w-8 h-8 text-gray-400 mx-auto" />
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Select watermark image</p>
                      <p className="text-sm text-gray-400 dark:text-gray-500">PNG recommended for transparency</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Settings Panel */}
          <div className="space-y-6">
            <div className="card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <AdjustmentsHorizontalIcon className="w-5 h-5 text-primary-600" />
                <h2 className="text-xl font-semibold">Watermark Settings</h2>
              </div>

              <div className="space-y-4">
                {/* Watermark Type Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Watermark Type
                  </label>
                  <div className="flex space-x-4">
                    <label className="inline-flex items-center">
                      <input 
                        type="radio" 
                        name="wm-type" 
                        value="visible" 
                        checked={watermarkType==='visible'} 
                        onChange={() => setWatermarkType('visible')} 
                        className="text-primary-600"
                      />
                      <span className="ml-2 text-gray-700 dark:text-gray-300">Visible</span>
                    </label>
                    <label className="inline-flex items-center">
                      <input 
                        type="radio" 
                        name="wm-type" 
                        value="invisible" 
                        checked={watermarkType==='invisible'} 
                        onChange={() => setWatermarkType('invisible')}
                        className="text-primary-600"
                      />
                      <span className="ml-2 text-gray-700 dark:text-gray-300">Invisible (DWT-DCT)</span>
                    </label>
                  </div>
                  {watermarkType === 'invisible' && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Invisible watermarks are embedded imperceptibly using DWT-DCT technique
                    </p>
                  )}
                </div>

                {/* Watermark Mode Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Watermark Content
                  </label>
                  <div className="flex space-x-4">
                    <label className="inline-flex items-center">
                      <input type="radio" name="wm-mode" value="image" checked={watermarkMode==='image'} onChange={() => setWatermarkMode('image')} />
                      <span className="ml-2 text-gray-700 dark:text-gray-300">Image</span>
                    </label>
                    <label className="inline-flex items-center">
                      <input type="radio" name="wm-mode" value="text" checked={watermarkMode==='text'} onChange={() => setWatermarkMode('text')} />
                      <span className="ml-2 text-gray-700 dark:text-gray-300">Text</span>
                    </label>
                  </div>
                </div>

                {watermarkMode === 'image' ? (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Watermark source
                    </label>
                    <input type="file" accept="image/*" onChange={(e)=> setWatermarkFile(e.target.files?.[0] || null)} />
                  </div>
                ) : (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Text watermark
                      </label>
                      <input className="input w-full" value={watermarkText} onChange={(e)=> setWatermarkText(e.target.value)} placeholder="Enter watermark text" />
                    </div>
                    <div className="flex space-x-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Text size
                        </label>
                        <input
                          type="text"
                          className="input w-24"
                          value={textSize}
                          onChange={e => {
                            let v = e.target.value;
                            // Allow empty for typing
                            if (v === '') {
                              setTextSize('');
                            } else {
                              // Only allow numbers
                              if (/^\d{0,3}$/.test(v)) {
                                setTextSize(v);
                              }
                            }
                          }}
                          onBlur={e => {
                            let num = parseInt(e.target.value, 10);
                            if (isNaN(num) || num < 30) setTextSize(30);
                            else if (num > 200) setTextSize(200);
                            else setTextSize(num);
                          }}
                          placeholder="30-200"
                          inputMode="numeric"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Color (name or hex)
                        </label>
                        <input className="input w-40" value={textColor} onChange={(e)=> setTextColor(e.target.value)} placeholder="#FFFFFF or red" />
                      </div>
                    </div>
                  </>
                )}

                {/* Invisible watermark specific settings */}
                {watermarkType === 'invisible' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Embedding Strength (Alpha): {invisibleAlpha.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.5"
                      step="0.01"
                      value={invisibleAlpha}
                      onChange={(e) => setInvisibleAlpha(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Higher values = stronger watermark but more visible. Default: 0.28
                    </p>
                  </div>
                )}

                {/* Visible watermark specific settings */}
                {watermarkType === 'visible' && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Position
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        <label className="inline-flex items-center"><input type="radio" name="pos" value="NW" checked={settings.position==='NW'} onChange={()=> setSettings({...settings, position:'NW'})} /><span className="ml-2 text-gray-700 dark:text-gray-300">Top Right</span></label>
                        <label className="inline-flex items-center"><input type="radio" name="pos" value="NE" checked={settings.position==='NE'} onChange={()=> setSettings({...settings, position:'NE'})} /><span className="ml-2 text-gray-700 dark:text-gray-300">Top Left</span></label>
                        <label className="inline-flex items-center"><input type="radio" name="pos" value="SW" checked={settings.position==='SW'} onChange={()=> setSettings({...settings, position:'SW'})} /><span className="ml-2 text-gray-700 dark:text-gray-300">Bottom Right</span></label>
                        <label className="inline-flex items-center"><input type="radio" name="pos" value="SE" checked={settings.position==='SE'} onChange={()=> setSettings({...settings, position:'SE'})} /><span className="ml-2 text-gray-700 dark:text-gray-300">Bottom Left</span></label>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Padding X: {settings.paddingX}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="50"
                      value={settings.paddingX}
                      onChange={(e) => setSettings({...settings, paddingX: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Padding Y: {settings.paddingY}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="50"
                      value={settings.paddingY}
                      onChange={(e) => setSettings({...settings, paddingY: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* NEW: Opacity control */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Opacity: {settings.opacity}%
                  </label>
                  <div className="flex items-center space-x-3">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={settings.opacity}
                      onChange={(e) =>
                        setSettings({ ...settings, opacity: parseInt(e.target.value || 0, 10) })
                      }
                      className="w-full"
                    />
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={settings.opacity}
                      onChange={(e) => {
                        const v = Math.max(0, Math.min(100, parseInt(e.target.value || 0, 10)));
                        setSettings({ ...settings, opacity: v });
                      }}
                      className="input w-20"
                    />
                    <span className="text-sm text-gray-500 dark:text-gray-400">%</span>
                  </div>
                </div>

                {/* NEW: Rotation control */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Rotation: {settings.rotation}°
                  </label>
                  <div className="flex items-center space-x-3">
                    <input
                      type="range"
                      min="-180"
                      max="180"
                      value={settings.rotation}
                      onChange={(e) =>
                        setSettings({ ...settings, rotation: parseInt(e.target.value || 0, 10) })
                      }
                      className="w-full"
                    />
                    <input
                      type="number"
                      min="-180"
                      max="180"
                      value={settings.rotation}
                      onChange={(e) => {
                        const v = Math.max(-180, Math.min(180, parseInt(e.target.value || 0, 10)));
                        setSettings({ ...settings, rotation: v });
                      }}
                      className="input w-20"
                    />
                    <span className="text-sm text-gray-500 dark:text-gray-400">°</span>
                  </div>
                  <div className="mt-2 flex gap-2">
                    <button
                      type="button"
                      onClick={() => setSettings({ ...settings, rotation: 0 })}
                      className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      0°
                    </button>
                    <button
                      type="button"
                      onClick={() => setSettings({ ...settings, rotation: 45 })}
                      className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      45°
                    </button>
                    <button
                      type="button"
                      onClick={() => setSettings({ ...settings, rotation: 90 })}
                      className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      90°
                    </button>
                    <button
                      type="button"
                      onClick={() => setSettings({ ...settings, rotation: -45 })}
                      className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      -45°
                    </button>
                  </div>
                  <p className="mt-1 text-xs text-gray-500">
                    Positive values rotate clockwise, negative values counter-clockwise
                  </p>
                </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="autoResize"
                        checked={settings.autoResize}
                        onChange={(e) => setSettings({...settings, autoResize: e.target.checked})}
                        className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                      />
                      <label htmlFor="autoResize" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                        Auto-resize watermark
                      </label>
                    </div>
                  </>
                )}
              </div>

              <button
                onClick={handleProcessClick}
                // Enable button in text mode even without a watermark image/file
                disabled={
                  isProcessing ||
                  images.length === 0 ||
                  (watermarkMode === 'image' && !watermarkImage && !watermarkFile)
                }
                className="btn-primary w-full mt-6 flex items-center justify-center space-x-2"
              >
                {isProcessing ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <PencilSquareIcon className="w-5 h-5" />
                    <span>Apply Watermarks</span>
                  </>
                )}
              </button>

              {/* Test Robustness Button - Only show for invisible watermarks */}
              {watermarkType === 'invisible' && (
                <div className="flex flex-col gap-2 mt-3">
                  <button
                    onClick={handleTestRobustness}
                    disabled={
                      isTestingRobustness ||
                      images.length === 0 ||
                      (watermarkMode === 'image' && !watermarkImage && !watermarkFile)
                    }
                    className="btn-secondary w-full flex items-center justify-center space-x-2"
                  >
                    {isTestingRobustness ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Testing...</span>
                      </>
                    ) : (
                      <>
                        <AdjustmentsHorizontalIcon className="w-5 h-5" />
                        <span>Test Robustness</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => setShowExtractPopup(true)}
                    className="btn-primary w-full mt-2 flex items-center justify-center space-x-2"
                  >
                    <span>Extract Watermark</span>
                  </button>
                </div>
              )}
  {/* Extract Watermark Popup */}
  <ExtractWatermarkPopup isOpen={showExtractPopup} onClose={() => setShowExtractPopup(false)} />
            </div>
          </div>

          {/* NEW: Live Preview Panel */}
          <div className="space-y-6">
            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Live Preview</h2>
                <span className="text-xs text-gray-500">
                  {previewLoading ? 'Rendering...' : (previewGrid.length > 0 ? `${previewGrid.length} preview(s)` : 'No preview')}
                </span>
              </div>

              {previewGrid.length === 0 && !previewLoading ? (
                <div className="text-sm text-gray-500">
                  {images.length === 0
                    ? 'Add images to see preview.'
                    : watermarkMode === 'image' && !watermarkFile && !watermarkImage
                    ? 'Select a watermark image to preview.'
                    : 'Adjust settings to see changes.'}
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {previewGrid.map((p) => (
                    <div key={p.id} className="rounded-lg overflow-hidden border border-gray-200">
                      <img src={p.url} alt={p.name} className="w-full object-cover" />
                      <div className="px-2 py-1 text-xs text-gray-600 truncate">{p.name}</div>
                    </div>
                  ))}
                </div>
              )}

              <p className="mt-3 text-xs text-gray-400">
                Preview is rendered by the server using the exact same algorithm as final processing.
              </p>
            </div>
          </div>
        </div>

        {/* Processed Images */}
        {processedImages.length > 0 && (
          <div className="mt-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                Processed Images ({processedImages.length})
              </h2>
              <button
                onClick={downloadAll}
                className="btn-primary flex items-center space-x-2"
              >
                <DocumentArrowDownIcon className="w-5 h-5" />
                <span>Download All</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {processedImages.map((image) => (
                <div key={image.id} className="card p-4">
                  <div className="relative group mb-3">
                    <img
                      src={image.preview}
                      alt={image.processedName}
                      className="w-full aspect-square object-cover rounded-lg"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                      <button className="p-2 bg-white bg-opacity-20 rounded-full text-white hover:bg-opacity-30 transition-colors">
                        <EyeIcon className="w-6 h-6" />
                      </button>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {image.processedName}
                    </p>
                    <p className="text-xs text-gray-500">
                      Original: {image.originalName}
                    </p>
                    <button
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = image.downloadUrl;
                        link.download = image.processedName;
                        link.click();
                      }}
                      className="btn-secondary w-full text-sm flex items-center justify-center space-x-1"
                    >
                      <DocumentArrowDownIcon className="w-4 h-4" />
                      <span>Download</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Robustness Test Results Modal */}
        {showRobustnessModal && robustnessResults && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-2xl font-semibold">Watermark Robustness Test Results</h2>
                  <button
                    onClick={() => setShowRobustnessModal(false)}
                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                <div className="mb-6">
                  <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Image:</span>
                      <span className="ml-2 font-medium">{robustnessResults.image_name}</span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Watermark Size:</span>
                      <span className="ml-2 font-medium">{robustnessResults.watermark_size}×{robustnessResults.watermark_size}</span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Embedding Strength (α):</span>
                      <span className="ml-2 font-medium">{robustnessResults.alpha}</span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Redundancy:</span>
                      <span className="ml-2 font-medium">{robustnessResults.redundancy}</span>
                    </div>
                    {/* NEW: Imperceptibility PSNR */}
                    <div className="col-span-2">
                      <span className="text-gray-600 dark:text-gray-400">Imperceptibility PSNR:</span>
                      <span className="ml-2 font-medium text-green-600 dark:text-green-400">
                        {robustnessResults.imperceptibility_psnr != null
                          ? `${Number(robustnessResults.imperceptibility_psnr).toFixed(2)} dB`
                          : 'N/A'}
                      </span>
                      <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">
                        (Original vs Watermarked)
                      </span>
                    </div>
                  </div>

                  {robustnessResults.original_image && (
                    <div className="mb-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Watermarked Image:</p>
                      <img 
                        src={robustnessResults.original_image} 
                        alt="Watermarked" 
                        className="max-w-full h-auto rounded border border-gray-300 dark:border-gray-600"
                        style={{maxHeight: '200px', objectFit: 'contain'}}
                      />
                    </div>
                  )}
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold mb-3">Attack Test Results</h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                      <thead className="bg-gray-50 dark:bg-gray-900">
                        <tr>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            Attack Type
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            PSNR (dB)
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            NCC Score
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            Status
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                        {robustnessResults.results.map((result, idx) => (
                          <tr key={idx} className={result.success ? '' : 'bg-red-50 dark:bg-red-900/20'}>
                            <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                              {result.attack}
                            </td>
                            <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                              {result.attack === 'Original (No Attack)' ? 'N/A' : (
                                <span className={
                                  result.psnr > 40 ? 'text-green-600 dark:text-green-400' :
                                  result.psnr > 30 ? 'text-yellow-600 dark:text-yellow-400' :
                                  'text-red-600 dark:text-red-400'
                                }>
                                  {result.psnr.toFixed(2)}
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                              <span className={
                                result.ncc > 0.8 ? 'text-green-600 dark:text-green-400' :
                                result.ncc > 0.6 ? 'text-yellow-600 dark:text-yellow-400' :
                                'text-red-600 dark:text-red-400'
                              }>
                                {result.ncc.toFixed(4)}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-sm">
                              {result.success ? (
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                                  ✓ Passed
                                </span>
                              ) : (
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                                  ✗ Failed
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h4 className="text-sm font-semibold mb-2 text-blue-900 dark:text-blue-100">Understanding the Results:</h4>
                  <ul className="text-xs text-blue-800 dark:text-blue-200 space-y-1">
                    {/* NEW: Add imperceptibility PSNR explanation */}
                    <li><strong>Imperceptibility PSNR:</strong> Measures how invisible the watermark is. &gt;40 dB = Excellent (invisible), 35-40 dB = Good, &lt;35 dB = Visible</li>
                    <li><strong>Attack PSNR (Peak Signal-to-Noise Ratio):</strong> Measures image quality after attack. Higher is better. &gt;40 dB = Excellent, 30-40 dB = Good, &lt;30 dB = Poor</li>
                    <li><strong>NCC (Normalized Cross-Correlation):</strong> Measures watermark detectability. &gt;0.8 = Strong, 0.6-0.8 = Moderate, &lt;0.6 = Weak/Failed</li>
                    <li><strong>Success Threshold:</strong> NCC &gt; 0.6 indicates the watermark is still detectable after the attack</li>
                  </ul>
                </div>

                <div className="mt-6 flex justify-end">
                  <button
                    onClick={() => setShowRobustnessModal(false)}
                    className="btn-secondary px-6 py-2"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Watermark;