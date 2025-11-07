import React, { useState, useEffect } from 'react';
import { PhotoIcon, SparklesIcon, AdjustmentsHorizontalIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { useAuth } from '../contexts/AuthContext';

const TextToImage = () => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [serverStatus, setServerStatus] = useState({ status: 'unknown', message: 'Checking server...' });
  const [settings, setSettings] = useState({
    model: 'df-gan',
    dataset: 'bird',
    batchSize: 1,
    seed: -1
  });
  const [showEvaluationDialog, setShowEvaluationDialog] = useState(false);
  const [pendingEvaluationData, setPendingEvaluationData] = useState(null);
  // NEW: selection mode for watermarking
  const [selectingForWatermark, setSelectingForWatermark] = useState(false);
  const [selectedForWatermark, setSelectedForWatermark] = useState(new Set());
  // NEW: selection mode for download
  const [selectingForDownload, setSelectingForDownload] = useState(false);
  const [selectedForDownload, setSelectedForDownload] = useState(new Set());
  const [showBatchSizeWarning, setShowBatchSizeWarning] = useState(false);
  const navigate = useNavigate();
  const { requireAuth } = useAuth();

  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5001';

  const datasets = [
    { value: 'bird', label: 'Birds (CUB-200-2011)' },
    { value: 'coco', label: 'COCO Dataset' },
    
  ];

  const examplePrompts = {
    bird: [
      "this bird is white with brown and has a very short beak.",
      "this bird has an orange bill, a white belly and white eyebrows.",
      "A blue jay with distinctive crest feathers",
      "this bird is white with red and has a very short beak."
    ],
    coco: [
      "A boat in the middle of the ocean.",
      "A large construction site for a bridge build.  ",
      "A kitchen has white counters and a wooden floor.",
      "On the plate is eggs,tomatoes sausage, and some bacon."
    ],
  };

  // Check server status on component mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/check`);
        const data = await res.json();
        
        if (res.ok) {
          setServerStatus({
            status: 'ready',
            message: 'Server is ready'
          });
        } else {
          setServerStatus({
            status: 'error',
            message: `Server issues: ${data.issues?.join(', ')}`,
            details: data
          });
        }
      } catch (error) {
        setServerStatus({
          status: 'error',
          message: 'Cannot connect to server. Is it running?'
        });
      }
    };
    
    checkServer();
  }, [API_BASE]);

  const handleGenerateClick = () => {
    requireAuth(handleGenerate);
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a text prompt');
      return;
    }

    setIsGenerating(true);
    try {
      // Generate a unique seed for each image in the batch
      const seeds = Array.from({ length: settings.batchSize }, () => Math.floor(Math.random() * 1000000));
      
      // Get token for authenticated requests (optional - saves to gallery if logged in)
      const token = localStorage.getItem('token');
      const headers = {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` })
      };
      
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
          prompt: prompt.trim(),
          dataset: settings.dataset, // 'bird' -> CUB, 'coco' -> COCO
          batchSize: settings.batchSize,
          seeds: seeds
        })
      });

      if (!res.ok) {
        // Prefer JSON error if provided by backend
        let msg = 'Generation failed';
        try {
          const j = await res.json();
          msg = j?.error || msg;
        } catch {
          const text = await res.text();
          msg = text || msg;
        }
        throw new Error(msg);
      }

      const data = await res.json();
      const images = (data.images || []).map((b64, i) => ({
        id: Date.now() + i,
        url: `data:image/png;base64,${b64}`,
        prompt,
        settings: { ...settings, seed: seeds[i] },
        createdAt: new Date()
      }));

      if (!images.length) throw new Error('No images returned from DF-GAN');

      setGeneratedImages(prev => [...images, ...prev]);
      toast.success(`Generated ${images.length} image(s) successfully!`);
      
      // Auto-save to gallery if user is authenticated
      const authToken = localStorage.getItem('ai_image_suite_auth_token');
      if (authToken) {
        // Save all generated images to gallery
        images.forEach(async (image) => {
          try {
            const base64Data = image.url.split(',')[1];
            await fetch(`${API_BASE}/api/gallery/save-generated`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
              },
              body: JSON.stringify({
                imageData: base64Data,
                prompt: image.prompt,
                dataset: image.settings.dataset
              })
            });
          } catch (error) {
            console.error('Failed to auto-save image to gallery:', error);
          }
        });
      }
      
      // NEW: Trigger auto-evaluation suggestion after 1.5 seconds
      setTimeout(() => {
        setPendingEvaluationData({
          prompt: prompt.trim(),
          images: images,
          timestamp: Date.now()
        });
        setShowEvaluationDialog(true);
      }, 1500);
      
    } catch (error) {
      toast.error(error.message || 'Failed to generate images. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  // NEW: Handle auto-evaluation acceptance
  const handleAutoEvaluationAccept = () => {
    if (pendingEvaluationData) {
      // Store the evaluation data globally for the ImageEvaluator to pick up
      sessionStorage.setItem('autoEvaluationData', JSON.stringify({
        prompt: pendingEvaluationData.prompt,
        images: pendingEvaluationData.images.map(img => ({
          id: img.id,
          url: img.url,
          prompt: img.prompt
        })),
        timestamp: pendingEvaluationData.timestamp
      }));
      
      // Navigate to evaluation page
      navigate('/evaluate');
    }
    setShowEvaluationDialog(false);
    setPendingEvaluationData(null);
  };

  // NEW: Handle auto-evaluation decline
  const handleAutoEvaluationDecline = () => {
    setShowEvaluationDialog(false);
    setPendingEvaluationData(null);
  };

  // NEW: start selection mode for watermarking
  const startWatermarkSelection = () => {
    setShowEvaluationDialog(false);
    setSelectingForWatermark(true);
    setSelectedForWatermark(new Set());
  };

  // NEW: toggle image selection
  const toggleSelectForWatermark = (id) => {
    setSelectedForWatermark(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  // NEW: send selected images to Watermark page
  const sendSelectedToWatermark = () => {
    const selected = generatedImages.filter(img => selectedForWatermark.has(img.id));
    if (selected.length === 0) {
      toast.error('Select at least one image');
      return;
    }
    sessionStorage.setItem('watermarkSelectionData', JSON.stringify({
      images: selected.map(img => ({
        url: img.url,
        name: `generated-${img.id}.png`
      }))
    }));
    setSelectingForWatermark(false);
    setSelectedForWatermark(new Set());
    navigate('/watermark');
  };

  // NEW: cancel selection mode
  const cancelWatermarkSelection = () => {
    setSelectingForWatermark(false);
    setSelectedForWatermark(new Set());
  };

  // NEW: start selection mode for downloading (triggered from dialog)
  const startDownloadSelection = () => {
    setShowEvaluationDialog(false);
    setSelectingForWatermark(false);
    setSelectedForWatermark(new Set());
    setSelectingForDownload(true);
    setSelectedForDownload(new Set());
  };

  // NEW: toggle/select helpers for download mode
  const toggleSelectForDownload = (id) => {
    setSelectedForDownload(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };
  const selectAllForDownload = () => {
    setSelectedForDownload(new Set(generatedImages.map(img => img.id)));
  };
  const cancelDownloadSelection = () => {
    setSelectingForDownload(false);
    setSelectedForDownload(new Set());
  };
  const downloadSelectedImages = () => {
    const selected = generatedImages.filter(img => selectedForDownload.has(img.id));
    if (selected.length === 0) {
      toast.error('Select at least one image to download');
      return;
    }
    selected.forEach(img => {
      const link = document.createElement('a');
      link.href = img.url;
      link.download = `generated-${img.id}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
    toast.success(`Downloading ${selected.length} image(s)`);
  };

  // Helpers to drive unified selection UI on the grid
  const isAnySelecting = selectingForWatermark || selectingForDownload;
  const isSelected = (id) => {
    return selectingForWatermark
      ? selectedForWatermark.has(id)
      : selectingForDownload
      ? selectedForDownload.has(id)
      : false;
  };
  const toggleSelectActive = (id) => {
    if (selectingForWatermark) toggleSelectForWatermark(id);
    else if (selectingForDownload) toggleSelectForDownload(id);
  };

  const setExamplePrompt = (examplePrompt) => {
    setPrompt(examplePrompt);
  };

  // NEW: Select all images for watermarking (CLIP dialog flow)
  const selectAllForWatermark = () => {
    setSelectedForWatermark(new Set(generatedImages.map(img => img.id)));
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Text-to-Image Generation
          </h1>
          <p className="text-gray-600 dark:text-gray-300 text-lg max-w-2xl mx-auto">
            Create stunning images from text descriptions using the DF-GAN model.
            Enter your prompt and customize the generation settings below.
          </p>
          
          {/* Server Status Indicator */}
          <div className={`mt-4 inline-flex items-center px-4 py-2 rounded-full text-sm ${
            serverStatus.status === 'ready' 
              ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300' 
              : serverStatus.status === 'error'
                ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300'
          }`}>
            <div className={`w-2 h-2 rounded-full mr-2 ${
              serverStatus.status === 'ready' 
                ? 'bg-green-500' 
                : serverStatus.status === 'error'
                  ? 'bg-red-500'
                  : 'bg-yellow-500'
            }`}></div>
            {serverStatus.message}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Prompt Input */}
            <div className="card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <SparklesIcon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Prompt</h2>
              </div>
              
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe the image you want to generate..."
                className="textarea h-32 mb-4"
                disabled={isGenerating}
              />

              {/* Disclaimer */}
              <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <div className="flex items-start space-x-2">
                  <span className="text-blue-600 dark:text-blue-400 text-lg flex-shrink-0">💡</span>
                  <div>
                    <p className="text-sm text-blue-800 dark:text-blue-300 font-medium mb-2">
                      COCO Dataset - What Works Best:
                    </p>
                    <div className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                      <p><strong>✅ Full Scene Descriptions:</strong> "A kitchen with white counters and wooden floor", "Train on tracks with water tower"</p>
                      <p><strong>✅ Individual Objects:</strong> Vehicles (car, motorcycle, truck, bus, train etc), Animals (cat, dog, tiger, lion, bear, wolf, leopard, cheetah etc)</p>
                      <p><strong>✅ Limited Food Items:</strong> Fruits (lemon, apple, orange, banana etc), "eggs and bacon on plate", "cupcakes", "donuts" etc </p>
  
                     
                    </div>
                  </div>
                </div>
              </div>

              {/* Example Prompts */}
              <div className="mb-4">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Example prompts for {datasets.find(d => d.value === settings.dataset)?.label}:
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {examplePrompts[settings.dataset]?.map((example, index) => (
                    <button
                      key={index}
                      onClick={() => setExamplePrompt(example)}
                      className="text-left p-2 text-sm bg-gray-50 dark:bg-gray-700 hover:bg-primary-50 dark:hover:bg-primary-900/30 hover:text-primary-700 dark:hover:text-primary-400 rounded border border-gray-200 dark:border-gray-600 text-gray-600 dark:text-gray-300 transition-colors"
                      disabled={isGenerating}
                    >
                      "{example}"
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={handleGenerateClick}
                disabled={isGenerating || !prompt.trim()}
                className="btn-primary w-full flex items-center justify-center space-x-2"
              >
                {isGenerating ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span className="loading-dots">Generating</span>
                  </>
                ) : (
                  <>
                    <PhotoIcon className="w-5 h-5" />
                    <span>Generate Images</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Settings Panel */}
          <div className="space-y-6">
            <div className="card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <AdjustmentsHorizontalIcon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Settings</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Dataset
                  </label>
                  <select
                    value={settings.dataset}
                    onChange={(e) => setSettings({...settings, dataset: e.target.value})}
                    className="input"
                    disabled={isGenerating}
                  >
                    {datasets.map(dataset => (
                      <option key={dataset.value} value={dataset.value}>
                        {dataset.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Number of Images
                </label>
                <input
                  type="text"
                  value={settings.batchSize}
                  onChange={(e) => {
                    const value = e.target.value;

                    // NEW: Allow empty string for typing.
                    // The onBlur handler will fix it if left empty.
                    if (value === '') {
                      setSettings({...settings, batchSize: ''});
                      setShowBatchSizeWarning(false);
                      return;
                    }

                    // Check for valid digits
                    if (/^\d+$/.test(value)) {
                      const numValue = parseInt(value, 10);
                      
                      if (numValue > 50) {
                        // Clamp at 50
                        setSettings({...settings, batchSize: 50});
                        setShowBatchSizeWarning(true);
                        setTimeout(() => setShowBatchSizeWarning(false), 3000);
                      } else {
                        // Store any other valid number (including 0 or 1)
                        // The onBlur handler will fix 0
                        setSettings({...settings, batchSize: numValue});
                        setShowBatchSizeWarning(false);
                      }
                    }
                    // If not a digit (e.g., 'a', '1.5'), do nothing
                  }}
                  onBlur={(e) => {
                    // Ensure valid number on blur
                    // This logic is now correct, as it handles the empty string or 0
                    if (e.target.value === '' || parseInt(e.target.value, 10) < 1) {
                      setSettings({...settings, batchSize: 1});
                      setShowBatchSizeWarning(false);
                    }
                  }}
                  placeholder="1-50"
                  className="input"
                  disabled={isGenerating}
                />
                {showBatchSizeWarning && (
                  <p className="mt-2 text-sm text-red-600 dark:text-red-400 animate-fade-in">
                    ⚠️ Please enter a number between 1 and 50
                  </p>
                )}
              </div>

                <div>
                  {/* Seed input removed: seeds are now auto-generated per image */}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Generated Images */}
        {generatedImages.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Generated Images</h2>
              <div className="flex items-center space-x-3">
                {/* NEW: watermark selection toolbar (unchanged) */}
                {selectingForWatermark && (
                  <>
                    {/* NEW: Select All button for watermark selection */}
                    <button
                      className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition"
                      onClick={selectAllForWatermark}
                    >
                      Select All
                    </button>
                    <button
                      className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 transition"
                      onClick={sendSelectedToWatermark}
                    >
                      Send to Watermark ({selectedForWatermark.size})
                    </button>
                    <button
                      className="px-4 py-2 bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-200 rounded hover:bg-gray-300 dark:hover:bg-gray-500 transition"
                      onClick={cancelWatermarkSelection}
                    >
                      Cancel
                    </button>
                  </>
                )}
                {/* NEW: download selection toolbar */}
                {selectingForDownload && (
                  <>
                    <button
                      className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition"
                      onClick={selectAllForDownload}
                    >
                      Select All
                    </button>
                    <button
                      className="px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 transition"
                      onClick={downloadSelectedImages}
                    >
                      Download {selectedForDownload.size} selected image{selectedForDownload.size === 1 ? '' : 's'}
                    </button>
                    <button
                      className="px-4 py-2 bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-200 rounded hover:bg-gray-300 dark:hover:bg-gray-500 transition"
                      onClick={cancelDownloadSelection}
                    >
                      Cancel
                    </button>
                  </>
                )}
                {/* ...existing code... */}
                {!isAnySelecting && (
                  <button
                    className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition"
                    onClick={() => setGeneratedImages([])}
                  >
                    Clear Images
                  </button>
                )}
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {generatedImages.map((image) => (
                <div key={image.id} className="card p-4 image-fade-in relative">
                  {/* ...existing code... */}
                  <img
                    src={image.url}
                    alt={image.prompt}
                    className={`w-full aspect-square object-cover rounded-lg mb-3 ${isAnySelecting && isSelected(image.id) ? 'ring-4 ring-primary-500' : ''}`}
                  />
                  {/* NEW: selection checkbox overlay for both modes */}
                  {isAnySelecting && (
                    <button
                      onClick={() => toggleSelectActive(image.id)}
                      className="absolute top-3 left-3 w-6 h-6 rounded bg-white border-2 border-gray-300 flex items-center justify-center"
                      title="Select image"
                    >
                      {isSelected(image.id) && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-primary-600" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-7.25 7.25a1 1 0 01-1.414 0l-3-3a1 1 0 111.414-1.414l2.293 2.293 6.543-6.543a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </button>
                  )}
                  {/* ...existing code... */}
                  <p className="text-sm text-gray-600 line-clamp-2 mb-2">
                    "{image.prompt}"
                  </p>
                  <div className="flex justify-between items-center text-xs text-gray-500">
                    <span>{image.settings.dataset}</span>
                    <span>{image.createdAt.toLocaleTimeString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* NEW: Auto-Evaluation Suggestion Dialog */}
      {showEvaluationDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <SparklesIcon className="w-6 h-6 text-primary-600" />
                <h3 className="text-lg font-semibold text-gray-900">
                  Try CLIP-based Analysis
                </h3>
              </div>
              <button
                onClick={handleAutoEvaluationDecline}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            
            <div className="mb-6">
              <p className="text-gray-600 mb-3">
                Great! You've generated {pendingEvaluationData?.images?.length || 0} image(s). 
                Would you like to automatically evaluate how well they match your prompt using our 
                advanced CLIP-based analysis?
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-sm text-blue-800">
                  <strong>What you'll get:</strong>
                  <br />• Semantic similarity scores
                  <br />• Detailed keyword analysis  
                  <br />• Feature presence detection
                  <br />• Quality recommendations
                </p>
              </div>
            </div>
            
            {/* UPDATED: responsive buttons with better wrapping */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                // CHANGED: start download selection instead of simple decline
                onClick={startDownloadSelection}
                className="w-full px-4 py-3 border border-gray-300 bg-white text-gray-800 rounded-lg hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm md:text-base text-left whitespace-normal break-words leading-snug"
              >
                No, Directly select and download generated images
              </button>
              <button
                onClick={handleAutoEvaluationAccept}
                className="w-full px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm md:text-base text-left whitespace-normal break-words leading-snug"
              >
                Yes, Analyze!
              </button>
            </div>
            {/* UPDATED: third action button styling */}
            <div className="mt-3">
              <button
                onClick={startWatermarkSelection}
                className="w-full px-4 py-3 bg-gray-100 text-gray-800 rounded-lg hover:bg-gray-200 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm md:text-base text-left whitespace-normal break-words leading-snug"
              >
                No, Directly select Images for watermarking
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TextToImage;

