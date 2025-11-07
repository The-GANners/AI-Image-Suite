import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  CheckCircleIcon, 
  CloudArrowUpIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  StarIcon,
  PhotoIcon,
  ArrowPathIcon,
  PlayIcon,
  PauseIcon,
  TrophyIcon,
  ArrowDownTrayIcon,
  PaintBrushIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
// NEW: navigate to Watermark page after selection
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const ImageEvaluator = () => {
  const { requireAuth } = useAuth();
  const [image, setImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluation, setEvaluation] = useState(null);
  const [threshold, setThreshold] = useState(0.25);

  // NEW: Auto-evaluation state
  const [autoEvaluationImages, setAutoEvaluationImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [isAutoMode, setIsAutoMode] = useState(false);
  const [batchResults, setBatchResults] = useState([]);

  // NEW: Auto-evaluation enhancement state
  const [autoEvaluateEnabled, setAutoEvaluateEnabled] = useState(true);
  const [isAutoEvaluating, setIsAutoEvaluating] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  // NEW: post-evaluation dialog and selection flows
  const [showPostEvalDialog, setShowPostEvalDialog] = useState(false);
  const [selectingForWatermark, setSelectingForWatermark] = useState(false);
  const [selectedForWatermark, setSelectedForWatermark] = useState(new Set());
  const [selectingForDownload, setSelectingForDownload] = useState(false);
  const [selectedForDownload, setSelectedForDownload] = useState(new Set());
  // NEW: manual (non-auto) upload tracking and selection states
  const [manualUploadedImages, setManualUploadedImages] = useState([]);
  const [showManualPostEvalDialog, setShowManualPostEvalDialog] = useState(false);
  const [selectingManualForWatermark, setSelectingManualForWatermark] = useState(false);
  const [selectedManualForWatermark, setSelectedManualForWatermark] = useState(new Set());
  // NEW: manual batch navigation and per-image results
  const [manualIndex, setManualIndex] = useState(0);
  const [manualResults, setManualResults] = useState([]);
  // NEW: manual evaluating state
  const [isManualEvaluating, setIsManualEvaluating] = useState(false);

  const navigate = useNavigate();

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    // CHANGED: allow multi-upload and track manually uploaded images for later selection
    onDrop: (acceptedFiles) => {
      if (!acceptedFiles || acceptedFiles.length === 0) return;
      const newItems = acceptedFiles.map((file, idx) => ({
        id: Date.now() + idx,
        file,
        preview: URL.createObjectURL(file),
        url: URL.createObjectURL(file),
        name: file.name
      }));
      setManualUploadedImages(prev => [...prev, ...newItems]);
      // Maintain single-image evaluation behavior: use the first of this batch
      const file = acceptedFiles[0];
      setImage({ file, preview: URL.createObjectURL(file) });
      // NEW: reset manual navigation to the first image of the manual batch
      setManualIndex(0);
    },
    multiple: true
  });

  // NEW: Check for auto-evaluation data on component mount
  useEffect(() => {
    const autoData = sessionStorage.getItem('autoEvaluationData');
    if (autoData) {
      try {
        const data = JSON.parse(autoData);
        // Check if data is recent (within 5 minutes)
        if (Date.now() - data.timestamp < 5 * 60 * 1000) {
          setAutoEvaluationImages(data.images);
          setPrompt(data.prompt);
          setIsAutoMode(true);
          setCurrentImageIndex(0);
          
          // Set the first image
          if (data.images.length > 0) {
            const firstImage = data.images[0];
            // Convert data URL to blob for file handling
            fetch(firstImage.url)
              .then(res => res.blob())
              .then(blob => {
                setImage({
                  file: new File([blob], `generated_${firstImage.id}.png`, { type: 'image/png' }),
                  preview: firstImage.url
                });
                
                // NEW: Auto-trigger evaluation if enabled
                if (autoEvaluateEnabled) {
                  setTimeout(() => {
                    startAutoEvaluation();
                  }, 1000); // Small delay to ensure image is loaded
                }
              });
          }
          
          toast.success(`Auto-loaded ${data.images.length} images for evaluation!`);
        }
        // Clear the session storage
        sessionStorage.removeItem('autoEvaluationData');
      } catch (error) {
        console.error('Error loading auto-evaluation data:', error);
      }
    }
  }, [autoEvaluateEnabled]);

  // FIX: Evaluate current image with forced UI update for auto mode
  const evaluateCurrentImage = async (imageIndex = currentImageIndex, evaluationId = null, fileToEvaluate = null) => {
    const file = fileToEvaluate || image?.file;
    if (!file || !prompt.trim()) return;

    console.log(`🔍 Starting evaluation for Image ${imageIndex + 1} with ID: ${evaluationId}`);

    try {
      // Convert image to base64 from the provided file (not from stale state)
      const base64Image = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      const requestBody = {
        image: base64Image,
        prompt: prompt.trim(),
        threshold: threshold
      };
      
      // Call evaluation API
      const response = await fetch('/api/evaluate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Server returned non-JSON response');
      }

      const results = await response.json();
      if (!response.ok || results.error) {
        throw new Error(results.error || `HTTP ${response.status}`);
      }

      // Transform results with EXPLICIT evaluation ID
      const transformedResults = {
        prompt: results.prompt,
        overallScore: results.overall_score,
        percentageMatch: parseFloat(results.percentage_match.replace('%', '')),
        quality: results.quality,
        feedback: results.feedback,
        keywordAnalysis: (results.keyword_analysis || []).map(kw => ({
          keyword: kw.keyword,
          present: kw.status_type === 'present',
          confidence: parseFloat(kw.confidence.replace('%', '')) / 100,
          confidencePercent: kw.confidence,
          status: kw.status,
          statusType: kw.status_type,
          rawScore: kw.raw_score || 0
        })),
        contradictionWarning: results.contradiction_warning,
        missingFeatureAnalysis: results.missing_feature_analysis,
        detailedMetrics: results.detailed_metrics || { raw_score: 0, average_score: 0 },
        evaluationId: evaluationId,
        imageIndex: imageIndex,
        timestamp: Date.now()
      };

      console.log(`📊 Got FRESH results for Image ${imageIndex + 1} (ID: ${evaluationId}):`, {
        score: transformedResults.percentageMatch,
        quality: transformedResults.quality,
        evaluationId: transformedResults.evaluationId
      });

      // Update batch results FIRST
      if (isAutoMode) {
        const newResult = {
          imageIndex,
          imageId: autoEvaluationImages[imageIndex]?.id,
          result: transformedResults,
          evaluationId
        };
        setBatchResults(prev => {
          const filtered = prev.filter(r => r.imageIndex !== imageIndex);
          return [...filtered, newResult];
        });
      }

      // ALWAYS update evaluation with fresh results
      setEvaluation(transformedResults);
      console.log(`🎯 UI FORCE updated for Image ${imageIndex + 1} with score: ${transformedResults.percentageMatch}% (ID: ${evaluationId})`);
      
      // Auto-save to gallery if user is authenticated
      const authToken = localStorage.getItem('ai_image_suite_auth_token');
      if (authToken && image) {
        try {
          const base64Data = image.preview.split(',')[1];
          const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5001';
          await fetch(`${API_BASE}/api/gallery/save-evaluated`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${authToken}`
            },
            body: JSON.stringify({
              imageData: base64Data,
              originalName: image.file?.name || 'evaluated_image.png',
              prompt: prompt,
              score: transformedResults.percentageMatch
            })
          });
        } catch (error) {
          console.error('Failed to auto-save evaluated image to gallery:', error);
        }
      }
      
      // Return the results for the caller
      return transformedResults;
      
    } catch (error) {
      console.error(`❌ Evaluation failed for image ${imageIndex + 1}:`, error);
      throw error;
    }
  };

  // NEW: Add a key to force re-renders
  const [evaluationKey, setEvaluationKey] = useState(0);
  // NEW: Add a ref to track current evaluation to prevent stale results
  const [currentEvaluationId, setCurrentEvaluationId] = useState(null);

  // FIXED: Auto-evaluation sequence with explicit evaluation ID passing
  const startAutoEvaluation = async () => {
    if (!isAutoMode || autoEvaluationImages.length === 0) return;
    
    setIsAutoEvaluating(true);
    toast.success('Starting automatic evaluation of all images...');
    
    try {
      for (let i = 0; i < autoEvaluationImages.length; i++) {
        // Skip if already evaluated
        if (batchResults.some(r => r.imageIndex === i)) {
          continue;
        }
        
        console.log(`🚀 Starting evaluation for Image ${i + 1}`);
        
        // Generate unique evaluation ID to prevent stale results
        const evaluationId = `eval-${i}-${Date.now()}`;
        setCurrentEvaluationId(evaluationId);
        
        // Clear UI and set current index
        setEvaluation(null);
        setEvaluationKey(prev => prev + 1);
        setCurrentImageIndex(i);

        const currentImage = autoEvaluationImages[i];
        // Load the image and create a File we will pass to evaluation
        const blob = await fetch(currentImage.url).then(res => res.blob());
        const fileForEval = new File([blob], `generated_${currentImage.id}.png`, { type: 'image/png' });

        // Update preview for UI
        setImage({
          file: fileForEval,
          preview: currentImage.url
        });

        // Wait a bit to allow preview to render
        await new Promise(resolve => setTimeout(resolve, 300));

        // CRITICAL: pass fileForEval to avoid stale state
        try {
          const freshResults = await evaluateCurrentImage(i, evaluationId, fileForEval);
          setEvaluationKey(prev => prev + 1);
          console.log(`✅ Evaluation complete for Image ${i + 1}, score: ${freshResults.percentageMatch}%`);
        } catch (error) {
          console.error(`❌ Evaluation failed for Image ${i + 1}:`, error);
          toast.error(`Evaluation failed for Image ${i + 1}`);
        }

        // Small delay so user sees result
        await new Promise(resolve => setTimeout(resolve, 800));
      }
      
      // Generate recommendations after all evaluations
      generateRecommendations();
      toast.success('Automatic evaluation completed for all images!');
      // NEW: show post-evaluation action dialog
      setShowPostEvalDialog(true);
      
    } catch (error) {
      console.error('Auto-evaluation failed:', error);
      toast.error('Auto-evaluation failed. You can still evaluate manually.');
    } finally {
      setIsAutoEvaluating(false);
      setCurrentEvaluationId(null);
    }
  };

  // NEW: Exit auto mode
  const handleExitAutoMode = () => {
    setIsAutoMode(false);
    setAutoEvaluationImages([]);
    setCurrentImageIndex(0);
    setBatchResults([]);
    setRecommendations(null);
    setIsAutoEvaluating(false);
    setImage(null);
    setPrompt('');
    setEvaluation(null);
    toast.success('Exited auto-evaluation mode');
  };

  // NEW: Generate recommendations based on all evaluations
  const generateRecommendations = () => {
    if (batchResults.length === 0) return;
    
    // Sort by percentage match score
    const sortedResults = [...batchResults].sort((a, b) => 
      b.result.percentageMatch - a.result.percentageMatch
    );
    
    const bestResult = sortedResults[0];
    const worstResult = sortedResults[sortedResults.length - 1];
    
    // Calculate statistics
    const avgScore = batchResults.reduce((sum, r) => sum + r.result.percentageMatch, 0) / batchResults.length;
    const excellentCount = batchResults.filter(r => r.result.quality === 'Excellent').length;
    const goodCount = batchResults.filter(r => r.result.quality === 'Good').length;
    
    setRecommendations({
      bestImage: {
        index: bestResult.imageIndex,
        score: bestResult.result.percentageMatch,
        quality: bestResult.result.quality,
        imageUrl: autoEvaluationImages[bestResult.imageIndex]?.url
      },
      worstImage: {
        index: worstResult.imageIndex,
        score: worstResult.result.percentageMatch,
        quality: worstResult.result.quality
      },
      statistics: {
        avgScore: avgScore.toFixed(1),
        excellentCount,
        goodCount,
        totalEvaluated: batchResults.length
      },
      recommendation: bestResult.result.percentageMatch >= 80 
        ? `Image ${bestResult.imageIndex + 1} is excellent! Recommended for download.`
        : bestResult.result.percentageMatch >= 60
        ? `Image ${bestResult.imageIndex + 1} is the best option with good quality.`
        : `Consider regenerating - highest score is only ${bestResult.result.percentageMatch.toFixed(1)}%`
    });
  };

  // Enhanced handleEvaluate to work with manual evaluation
  const handleEvaluateClick = () => {
    requireAuth(handleEvaluate);
  };

  const handleEvaluate = async () => {
    console.log("🔍 [FRONTEND] Starting evaluation...");
    
    if (!image || !image.file || !prompt.trim()) {
      toast.error('Please upload an image and enter a prompt');
      return;
    }

    setIsEvaluating(true);
    
    try {
      // Test API connectivity first
      console.log("🔍 [FRONTEND] Testing API connectivity...");
      try {
        const testResponse = await fetch('/api/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ test: 'connectivity' })
        });
        
        if (!testResponse.ok) {
          throw new Error(`API test failed: ${testResponse.status}`);
        }
        
        const testData = await testResponse.json();
        console.log("✅ [FRONTEND] API connectivity test passed:", testData);
      } catch (testError) {
        console.error("❌ [FRONTEND] API connectivity test failed:", testError);
        throw new Error(`Cannot reach backend API: ${testError.message}`);
      }

      // NEW: pass proper index for manual vs auto
      const evalIndex = isAutoMode ? currentImageIndex : manualIndex;
      const fresh = await evaluateCurrentImage(evalIndex, null, image.file);
      // NEW: store per-image results for manual batch
      if (!isAutoMode && manualUploadedImages.length > 0) {
        setManualResults(prev => {
          const filtered = prev.filter(r => r.imageIndex !== manualIndex);
          return [...filtered, {
            imageIndex: manualIndex,
            imageId: manualUploadedImages[manualIndex]?.id,
            result: fresh
          }];
        });
      }
      
      console.log("✅ [FRONTEND] Results transformed successfully");
      toast.success('Evaluation completed successfully!');
      // NEW: If not in auto mode, show manual post-evaluation watermark dialog
      if (!isAutoMode) {
        setShowManualPostEvalDialog(true);
      }
    } catch (error) {
      console.error("❌ [FRONTEND] Evaluation failed:", error);
      toast.error(`Failed to evaluate: ${error.message}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  // NEW: Download recommended image
  const downloadRecommendedImage = () => {
    if (!recommendations?.bestImage?.imageUrl) return;
    
    const link = document.createElement('a');
    link.href = recommendations.bestImage.imageUrl;
    link.download = `best_image_${recommendations.bestImage.index + 1}_score_${recommendations.bestImage.score.toFixed(1)}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast.success('Downloading recommended image!');
  };

  const handleAddWatermark = () => {
    if (!image) {
      toast.error('No image to watermark');
      return;
    }

    // Store the evaluated image data for the Watermark page to pick up
    sessionStorage.setItem('watermarkSelectionData', JSON.stringify({
      images: [{
        url: image.preview,
        name: image.file?.name || 'evaluated_image.png',
        id: Date.now()
      }],
      source: 'evaluator'
    }));

    // Navigate to watermark page
    navigate('/watermark');
    toast.success('Image sent to watermark page!');
  };

  const getQualityColor = (quality) => {
    switch (quality) {
      case 'Excellent': return 'text-green-600 bg-green-100';
      case 'Good': return 'text-blue-600 bg-blue-100';
      case 'Fair': return 'text-yellow-600 bg-yellow-100';
      case 'Poor': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-blue-600';
    if (score >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  // FIXED: Navigation to prevent stale results
  const handleNextImage = () => {
    if (currentImageIndex < autoEvaluationImages.length - 1) {
      const nextIndex = currentImageIndex + 1;
      setCurrentImageIndex(nextIndex);
      setCurrentEvaluationId(null); // Clear evaluation ID during manual navigation
      const nextImage = autoEvaluationImages[nextIndex];
      
      fetch(nextImage.url)
        .then(res => res.blob())
        .then(blob => {
          setImage({
            file: new File([blob], `generated_${nextImage.id}.png`, { type: 'image/png' }),
            preview: nextImage.url
          });
          
          // Find the most recent result for this image
          const existingResult = batchResults.find(r => r.imageIndex === nextIndex);
          if (existingResult) {
            setEvaluation(existingResult.result);
            setEvaluationKey(prev => prev + 1);
            console.log(`🔄 Displaying STORED analysis for Image ${nextIndex + 1}:`, existingResult.result.percentageMatch);
            toast.success(`Showing analysis for Image ${nextIndex + 1} (${existingResult.result.percentageMatch.toFixed(1)}%)`);
          } else {
            setEvaluation(null);
            console.log(`❌ No analysis found for Image ${nextIndex + 1}`);
          }
        });
    }
  };

  const handlePreviousImage = () => {
    if (currentImageIndex > 0) {
      const prevIndex = currentImageIndex - 1;
      setCurrentImageIndex(prevIndex);
      setCurrentEvaluationId(null); // Clear evaluation ID during manual navigation
      const prevImage = autoEvaluationImages[prevIndex];
      
      fetch(prevImage.url)
        .then(res => res.blob())
        .then(blob => {
          setImage({
            file: new File([blob], `generated_${prevImage.id}.png`, { type: 'image/png' }),
            preview: prevImage.url
          });
          
          // Find the most recent result for this image
          const existingResult = batchResults.find(r => r.imageIndex === prevIndex);
          if (existingResult) {
            setEvaluation(existingResult.result);
            setEvaluationKey(prev => prev + 1);
            console.log(`🔄 Displaying STORED analysis for Image ${prevIndex + 1}:`, existingResult.result.percentageMatch);
            toast.success(`Showing analysis for Image ${prevIndex + 1} (${existingResult.result.percentageMatch.toFixed(1)}%)`);
          } else {
            setEvaluation(null);
            console.log(`❌ No analysis found for Image ${prevIndex + 1}`);
          }
        });
    }
  };

  // NEW: manual navigation helpers
  const loadManualImageAt = (idx) => {
    const item = manualUploadedImages[idx];
    if (!item) return;
    setImage({ file: item.file, preview: item.preview });
    const existing = manualResults.find(r => r.imageIndex === idx);
    if (existing) {
      setEvaluation(existing.result);
      setEvaluationKey(prev => prev + 1);
    } else {
      setEvaluation(null);
    }
  };

  const handleManualNextImage = () => {
    if (manualUploadedImages.length < 2) return;
    const nextIdx = Math.min(manualUploadedImages.length - 1, manualIndex + 1);
    if (nextIdx !== manualIndex) {
      setManualIndex(nextIdx);
      loadManualImageAt(nextIdx);
    }
  };

  const handleManualPreviousImage = () => {
    if (manualUploadedImages.length < 2) return;
    const prevIdx = Math.max(0, manualIndex - 1);
    if (prevIdx !== manualIndex) {
      setManualIndex(prevIdx);
      loadManualImageAt(prevIdx);
    }
  };

  // === Missing selection helpers (POST-EVAL Watermarking/Download) ===
  // Start selection mode for watermarking
  const startPostEvalWatermarkSelection = () => {
    setShowPostEvalDialog(false);
    setSelectingForDownload(false);
    setSelectedForDownload(new Set());
    setSelectingForWatermark(true);
    setSelectedForWatermark(new Set());
  };

  // Toggle selection for watermarking
  const toggleSelectForWatermark = (id) => {
    setSelectedForWatermark(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  // Cancel selection for watermarking
  const cancelWatermarkSelection = () => {
    setSelectingForWatermark(false);
    setSelectedForWatermark(new Set());
  };

  // Send selected images to Watermark page
  const sendSelectedToWatermark = () => {
    const selected = autoEvaluationImages.filter(img => selectedForWatermark.has(img.id));
    if (selected.length === 0) {
      toast.error('Select at least one image');
      return;
    }
    sessionStorage.setItem('watermarkSelectionData', JSON.stringify({
      images: selected.map(img => ({ url: img.url, name: `generated-${img.id}.png` }))
    }));
    setSelectingForWatermark(false);
    setSelectedForWatermark(new Set());
    navigate('/watermark');
  };

  // Start selection mode for downloading
  const startPostEvalDownloadSelection = () => {
    setShowPostEvalDialog(false);
    setSelectingForWatermark(false);
    setSelectedForWatermark(new Set());
    setSelectingForDownload(true);
    setSelectedForDownload(new Set());
  };

  // Toggle selection for downloading
  const toggleSelectForDownload = (id) => {
    setSelectedForDownload(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  // Select all images for downloading
  const selectAllForDownload = () => {
    setSelectedForDownload(new Set(autoEvaluationImages.map(img => img.id)));
  };

  // NEW: Select all images for watermarking (post-evaluation flow)
  const selectAllForWatermark = () => {
    setSelectedForWatermark(new Set(autoEvaluationImages.map(img => img.id)));
  };

  // Download selected images
  const downloadSelectedImages = () => {
    const selected = autoEvaluationImages.filter(img => selectedForDownload.has(img.id));
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

  // Cancel selection for downloading
  const cancelDownloadSelection = () => {
    setSelectingForDownload(false);
    setSelectedForDownload(new Set());
  };

  // === Manual (non-auto) watermark selection helpers ===
  // Derive the list of images eligible for manual selection (use uploaded list; fallback to current image)
  const manualSelectionList = manualUploadedImages && manualUploadedImages.length > 0
    ? manualUploadedImages
    : (image ? [{ id: Date.now(), preview: image.preview, url: image.preview, name: image.file?.name || 'uploaded.png' }] : []);

  const startManualWatermarkSelection = () => {
    setShowManualPostEvalDialog(false);
    setSelectingManualForWatermark(true);
    setSelectedManualForWatermark(new Set());
  };

  const toggleSelectManualForWatermark = (id) => {
    setSelectedManualForWatermark(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const selectAllManualForWatermark = () => {
    setSelectedManualForWatermark(new Set(manualSelectionList.map(img => img.id)));
  };

  const sendSelectedManualToWatermark = () => {
    const selected = manualSelectionList.filter(img => selectedManualForWatermark.has(img.id));
    if (selected.length === 0) {
      toast.error('Select at least one image');
      return;
    }
    sessionStorage.setItem('watermarkSelectionData', JSON.stringify({
      images: selected.map(img => ({
        url: img.preview || img.url,
        name: img.name || `uploaded-${img.id}.png`
      }))
    }));
    setSelectingManualForWatermark(false);
    setSelectedManualForWatermark(new Set());
    navigate('/watermark');
  };

  const cancelManualWatermarkSelection = () => {
    setSelectingManualForWatermark(false);
    setSelectedManualForWatermark(new Set());
  };

  // NEW: Evaluate all manually uploaded images (Manual Mode only)
  const startManualEvaluateAll = async () => {
    if (isAutoMode) return; // safety
    if (manualUploadedImages.length === 0) {
      toast.error('Upload images first');
      return;
    }
    if (!prompt.trim()) {
      toast.error('Enter a prompt to evaluate');
      return;
    }

    setIsManualEvaluating(true);
    toast.success('Starting evaluation of all uploaded images...');

    try {
      for (let i = 0; i < manualUploadedImages.length; i++) {
        // Skip if already evaluated
        if (manualResults.some(r => r.imageIndex === i)) continue;

        const item = manualUploadedImages[i];
        // Update UI preview and index
        setManualIndex(i);
        setImage({ file: item.file, preview: item.preview });

        // Small delay to let preview render
        await new Promise(resolve => setTimeout(resolve, 200));

        // Evaluate current file with a unique evaluationId
        const evalId = `manual-${i}-${Date.now()}`;
        try {
          const fresh = await evaluateCurrentImage(i, evalId, item.file);
          // Store per-image results for manual batch
          setManualResults(prev => {
            const filtered = prev.filter(r => r.imageIndex !== i);
            return [...filtered, { imageIndex: i, imageId: item.id, result: fresh }];
          });
          setEvaluationKey(prev => prev + 1);
        } catch (err) {
          console.error(`Manual evaluation failed for image ${i + 1}:`, err);
          toast.error(`Evaluation failed for image ${i + 1}`);
        }

        // Small delay so user sees result
        await new Promise(resolve => setTimeout(resolve, 300));
      }

      toast.success('Manual evaluation completed for all uploaded images!');
      // Show manual post-evaluation watermark dialog
      setShowManualPostEvalDialog(true);
    } catch (error) {
      console.error('Manual evaluate-all failed:', error);
      toast.error('Manual Evaluate All failed.');
    } finally {
      setIsManualEvaluating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Image Quality Evaluation
          </h1>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto dark:text-white">
            Analyze how well your AI-generated images match their text prompts using 
            advanced CLIP-based semantic understanding.
          </p>
          
          {/* Auto-mode indicator */}
          {isAutoMode && (
            <div className="mt-4 inline-flex items-center px-4 py-2 bg-blue-100 text-blue-800 rounded-full text-sm">
              <ArrowPathIcon className="w-4 h-4 mr-2" />
              Auto-Evaluation Mode: Image {currentImageIndex + 1} of {autoEvaluationImages.length}
              {/* NEW: Auto-evaluation controls */}
              <div className="ml-4 flex items-center space-x-2">
                {!isAutoEvaluating && batchResults.length < autoEvaluationImages.length && (
                  <button
                    onClick={startAutoEvaluation}
                    className="flex items-center space-x-1 px-2 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700"
                  >
                    <PlayIcon className="w-3 h-3" />
                    <span>Auto-Evaluate All</span>
                  </button>
                )}
                {isAutoEvaluating && (
                  <div className="flex items-center space-x-1 text-green-600">
                    <div className="animate-spin w-3 h-3 border border-green-600 border-t-transparent rounded-full"></div>
                    <span>Evaluating...</span>
                  </div>
                )}
              </div>
              <button
                onClick={handleExitAutoMode}
                className="ml-3 text-blue-600 hover:text-blue-800 underline"
              >
                Exit
              </button>
            </div>
          )}
        </div>

        {/* NEW: Recommendations Panel */}
        {recommendations && isAutoMode && (
          <div className="mb-8">
            <div className="card p-6 bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center space-x-2 text-yellow-800">
                  <TrophyIcon className="w-6 h-6" />
                  <span>Evaluation Complete - Recommendations</span>
                </h2>
                <button
                  onClick={downloadRecommendedImage}
                  className="flex items-center space-x-2 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-colors"
                >
                  <ArrowDownTrayIcon className="w-4 h-4" />
                  <span>Download Best</span>
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {recommendations.bestImage.score.toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">
                    Best Score (Image {recommendations.bestImage.index + 1})
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {recommendations.statistics.avgScore}%
                  </div>
                  <div className="text-sm text-gray-600">Average Score</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {recommendations.statistics.excellentCount + recommendations.statistics.goodCount}/{recommendations.statistics.totalEvaluated}
                  </div>
                  <div className="text-sm text-gray-600">Good+ Quality</div>
                </div>
              </div>
              
              <div className="bg-white rounded-lg p-4 border border-yellow-200">
                <p className="text-gray-700 font-medium">
                  📍 {recommendations.recommendation}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  Quality breakdown: {recommendations.statistics.excellentCount} Excellent, 
                  {' '}{recommendations.statistics.goodCount} Good, 
                  {' '}{recommendations.statistics.totalEvaluated - recommendations.statistics.excellentCount - recommendations.statistics.goodCount} Fair/Poor
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <div className="space-y-6">
            {/* Image Upload */}
            <div className="card p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <CloudArrowUpIcon className="w-5 h-5 text-primary-600" />
                <span>Upload Image</span>
                {/* Auto-mode navigation */}
                {isAutoMode && (
                  <div className="ml-auto flex items-center space-x-2">
                    <button
                      onClick={handlePreviousImage}
                      disabled={currentImageIndex === 0 || isAutoEvaluating}
                      className="p-1 rounded text-gray-400 hover:text-gray-600 disabled:opacity-50 transition-colors"
                      title={`Go to Image ${currentImageIndex} ${batchResults.find(r => r.imageIndex === currentImageIndex - 1) ? '(Analyzed)' : '(Not analyzed)'}`}
                    >
                      ←
                    </button>
                    <span className="text-xs text-gray-500">
                      {currentImageIndex + 1}/{autoEvaluationImages.length}
                    </span>
                    <button
                      onClick={handleNextImage}
                      disabled={currentImageIndex === autoEvaluationImages.length - 1 || isAutoEvaluating}
                      className="p-1 rounded text-gray-400 hover:text-gray-600 disabled:opacity-50 transition-colors"
                      title={`Go to Image ${currentImageIndex + 2} ${batchResults.find(r => r.imageIndex === currentImageIndex + 1) ? '(Analyzed)' : '(Not analyzed)'}`}
                    >
                      →
                    </button>
                  </div>
                )}
                {/* Manual navigation (non-auto mode) */}
                {!isAutoMode && manualUploadedImages.length > 1 && (
                  <div className="ml-auto flex items-center space-x-2">
                    <button
                      onClick={handleManualPreviousImage}
                      disabled={manualIndex === 0 || isEvaluating}
                      className="p-1 rounded text-gray-400 hover:text-gray-600 disabled:opacity-50 transition-colors"
                      title={`Go to Image ${manualIndex} ${manualResults.find(r => r.imageIndex === manualIndex - 1) ? '(Analyzed)' : '(Not analyzed)'}`}
                    >
                      ←
                    </button>
                    <span className="text-xs text-gray-500">
                      {manualIndex + 1}/{manualUploadedImages.length}
                    </span>
                    <button
                      onClick={handleManualNextImage}
                      disabled={manualIndex === manualUploadedImages.length - 1 || isEvaluating}
                      className="p-1 rounded text-gray-400 hover:text-gray-600 disabled:opacity-50 transition-colors"
                      title={`Go to Image ${manualIndex + 2} ${manualResults.find(r => r.imageIndex === manualIndex + 1) ? '(Analyzed)' : '(Not analyzed)'}`}
                    >
                      →
                    </button>
                  </div>
                )}
                {/* NEW: Manual Evaluate All controls */}
                {!isAutoMode && manualUploadedImages.length > 0 && (
                  <div className="ml-4 flex items-center space-x-2">
                    {!isManualEvaluating && manualResults.length < manualUploadedImages.length && (
                      <button
                        onClick={startManualEvaluateAll}
                        className="flex items-center space-x-1 px-2 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700"
                      >
                        <PlayIcon className="w-3 h-3" />
                        <span>Evaluate All</span>
                      </button>
                    )}
                    {isManualEvaluating && (
                      <div className="flex items-center space-x-1 text-green-600">
                        <div className="animate-spin w-3 h-3 border border-green-600 border-t-transparent rounded-full"></div>
                        <span>Evaluating...</span>
                      </div>
                    )}
                  </div>
                )}
              </h2>
              
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
                  isDragActive 
                    ? 'border-primary-400 bg-primary-50 dark:bg-primary-900' 
                    : 'border-gray-300 hover:border-primary-400 dark:border-gray-600'
                } ${isAutoMode ? 'border-blue-300 bg-blue-50 dark:bg-blue-900' : ''}`}
              >
                <input {...getInputProps()} />
                {image ? (
                  <div className="space-y-4">
                    <img 
                      src={image.preview} 
                      alt="Uploaded" 
                      className="max-h-64 mx-auto rounded-lg shadow-sm"
                    />
                    <p className="text-sm text-gray-600 dark:text-white">
                      {isAutoMode ? 'Auto-loaded from generation' : 'Click or drag to replace image'}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <CloudArrowUpIcon className="w-12 h-12 text-gray-400 dark:text-gray-300 mx-auto" />
                    <div>
                      <p className="text-lg text-gray-600 dark:text-white mb-2">
                        {isDragActive ? 'Drop your image here' : 'Drag & drop an image here'}
                      </p>
                      <p className="text-sm text-gray-400 dark:text-gray-200">
                        or click to select • PNG, JPG, JPEG, WebP
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Prompt Input */}
            <div className="card p-6">
              <h2 className="text-xl font-semibold mb-4">Text Prompt</h2>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter the text prompt used to generate this image..."
                className={`textarea h-32 mb-4 ${isAutoMode ? 'bg-blue-50 border-blue-200' : ''}`}
                disabled={isEvaluating}
              />
              
              {isAutoMode && (
                <p className="text-xs text-blue-600 mb-4">
                  ✨ Prompt auto-filled from generation
                </p>
              )}

              <button
                onClick={handleEvaluateClick}
                disabled={(isEvaluating || isAutoEvaluating) || !image || !prompt.trim()}
                className="btn-primary w-full flex items-center justify-center space-x-2"
              >
                {(isEvaluating || isAutoEvaluating) ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Evaluating...</span>
                  </>
                ) : (
                  <>
                    <CheckCircleIcon className="w-5 h-5" />
                    <span>Evaluate Image</span>
                  </>
                )}
              </button>
              
              {/* NEW: Auto-evaluation toggle */}
              {isAutoMode && (
                <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={autoEvaluateEnabled}
                      onChange={(e) => setAutoEvaluateEnabled(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-sm text-blue-800">
                      Enable automatic evaluation for all images
                    </span>
                  </label>
                </div>
              )}

              {/* Auto-mode navigation buttons */}
              {isAutoMode && autoEvaluationImages.length > 1 && (
                <div className="flex space-x-3 mt-4">
                  <button
                    onClick={handlePreviousImage}
                    disabled={currentImageIndex === 0 || isAutoEvaluating}
                    className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors"
                  >
                    ← Previous Image
                    {/* NEW: Show analysis status */}
                    {batchResults.find(r => r.imageIndex === currentImageIndex - 1) && (
                      <span className="ml-1 text-green-600">✓</span>
                    )}
                  </button>
                  <button
                    onClick={handleNextImage}
                    disabled={currentImageIndex === autoEvaluationImages.length - 1 || isAutoEvaluating}
                    className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors"
                  >
                    Next Image →
                    {/* NEW: Show analysis status */}
                    {batchResults.find(r => r.imageIndex === currentImageIndex + 1) && (
                      <span className="ml-1 text-green-600">✓</span>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-6" key={`evaluation-${evaluationKey}-${currentImageIndex}-${currentEvaluationId || 'nav'}`}>
            {evaluation ? (
              <>
                {/* Show current image info in results header */}
                {isAutoMode ? (
                  <div className="card p-3 bg-gray-50 border border-gray-200">
                    <p className="text-sm text-gray-600 text-center">
                      📊 Showing results for <strong>Image {currentImageIndex + 1}</strong> of {autoEvaluationImages.length}
                      {batchResults.find(r => r.imageIndex === currentImageIndex) && (
                        <span className="ml-2 text-green-600">✓ Analyzed</span>
                      )}
                      {/* Show unique evaluation info for debugging */}
                      <span className="ml-2 text-xs text-gray-400">
                        (eval #{evaluationKey} | {evaluation.evaluationId ? 'FRESH' : 'STORED'} | {evaluation.percentageMatch?.toFixed(1)}%)
                      </span>
                    </p>
                  </div>
                ) : (
                  // NEW: Manual-mode results header
                  manualUploadedImages.length > 0 && (
                    <div className="card p-3 bg-gray-50 border border-gray-200">
                      <p className="text-sm text-gray-600 text-center">
                        📊 Showing results for <strong>Manual Image {manualIndex + 1}</strong> of {manualUploadedImages.length}
                        {manualResults.find(r => r.imageIndex === manualIndex) && (
                          <span className="ml-2 text-green-600">✓ Analyzed</span>
                        )}
                        <span className="ml-2 text-xs text-gray-400">
                          (eval #{evaluationKey} | {evaluation.evaluationId ? 'FRESH' : 'STORED'} | {evaluation.percentageMatch?.toFixed(1)}%)
                        </span>
                      </p>
                    </div>
                  )
                )}

                {/* Batch results summary in auto mode */}
                {isAutoMode && batchResults.length > 0 && (
                  <div className="card p-4 bg-blue-50 border border-blue-200">
                    <h3 className="text-sm font-semibold text-blue-800 mb-2">
                      Batch Progress: {batchResults.length}/{autoEvaluationImages.length} evaluated
                      {recommendations && (
                        <span className="ml-2 text-green-600">✓ Complete</span>
                      )}
                    </h3>
                    <div className="flex space-x-1">
                      {autoEvaluationImages.map((_, index) => {
                        const result = batchResults.find(r => r.imageIndex === index);
                        const isRecommended = recommendations?.bestImage?.index === index;
                        const isCurrent = index === currentImageIndex;
                        return (
                          <div
                            key={index}
                            className={`w-4 h-2 rounded relative ${
                              result
                                ? result.result.percentageMatch >= 80
                                  ? 'bg-green-400'
                                  : result.result.percentageMatch >= 60
                                  ? 'bg-blue-400'
                                  : 'bg-yellow-400'
                                : isCurrent
                                ? 'bg-blue-400'
                                : 'bg-gray-300'
                            } ${isCurrent ? 'ring-2 ring-blue-600' : ''}`}
                            title={`Image ${index + 1}${result ? ` - ${result.result.percentageMatch.toFixed(1)}%` : ''}`}
                          >
                            {isRecommended && (
                              <div className="absolute -top-1 left-1/2 transform -translate-x-1/2">
                                <TrophyIcon className="w-2 h-2 text-yellow-600" />
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                
                {/* Overall Score */}
                <div className="card p-6" key={`score-${evaluationKey}-${evaluation.percentageMatch}-${evaluation.timestamp || Date.now()}`}>
                  <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                    <ChartBarIcon className="w-5 h-5 text-primary-600" />
                    <span>Module-2 Evaluation Results</span>
                    {/* Show result freshness indicator */}
                    {evaluation.evaluationId && (
                      <span className="text-xs text-green-600">FRESH</span>
                    )}
                  </h2>

                  <div className="text-center mb-6">
                    <div className={`text-6xl font-bold mb-2 ${getScoreColor(evaluation.percentageMatch)}`}>
                      {evaluation.percentageMatch.toFixed(2)}%
                    </div>
                    <div className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${getQualityColor(evaluation.quality)}`}>
                      {evaluation.quality}
                    </div>
                  </div>

                  <p className="text-gray-600 text-center mb-6">
                    {evaluation.feedback}
                  </p>

                  {/* Contradiction Warning */}
                  {evaluation.contradictionWarning && (
                    <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-800">
                      <ExclamationTriangleIcon className="w-4 h-4 inline mr-2" />
                      {evaluation.contradictionWarning}
                    </div>
                  )}

                  {/* Progress Bar */}
                  <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                    <div 
                      className={`h-3 rounded-full progress-bar ${
                        evaluation.percentageMatch >= 80 ? 'bg-green-500' :
                        evaluation.percentageMatch >= 60 ? 'bg-blue-500' :
                        evaluation.percentageMatch >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${Math.min(100, evaluation.percentageMatch)}%` }}
                    />
                  </div>

                  {/* Detailed Metrics */}
                  {evaluation.detailedMetrics && (
                    <div className="mt-4 text-xs text-gray-500 space-y-1">
                      <div className="flex justify-between">
                        <span className="dark:text-gray-200">Raw CLIP Score:</span>
                        <span className="font-mono dark:text-gray-200">{evaluation.detailedMetrics.raw_score?.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="dark:text-gray-200">Average Score:</span>
                        <span className="font-mono dark:text-gray-200">{evaluation.detailedMetrics.average_score?.toFixed(4)}</span>
                      </div>
                    </div>
                  )}
                  
                  {/* Action Buttons */}
                  <div className="mt-6">
                    <button
                      onClick={handleAddWatermark}
                      className="w-full px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors flex items-center justify-center space-x-2"
                      title="Add watermark to this image"
                    >
                      <PaintBrushIcon className="w-5 h-5" />
                      <span>Add Watermark</span>
                    </button>
                  </div>
                </div>

                {/* Keyword Analysis - ADD KEY PROP */}
                <div className="card p-6" key={`keywords-${evaluationKey}-${evaluation.keywordAnalysis.length}`}>
                  <h3 className="text-lg font-semibold mb-4 dark:text-gray-100">Detailed Keyword Analysis</h3>
                  <div className="space-y-3">
                    {evaluation.keywordAnalysis.map((keyword, index) => (
                      <div
                        key={`${evaluationKey}-${keyword.keyword}-${index}`}
                        className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                      >
                        <div className="flex items-center space-x-3">
                          {keyword.statusType === 'present' ? (
                            <CheckCircleIcon className="w-5 h-5 text-green-500 dark:text-green-400" />
                          ) : keyword.statusType === 'weak' ? (
                            <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500 dark:text-yellow-400" />
                          ) : (
                            <ExclamationTriangleIcon className="w-5 h-5 text-red-500 dark:text-red-400" />
                          )}
                          <div>
                            <span className="font-medium text-gray-800 dark:text-gray-100">{keyword.keyword}</span>
                            <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">{keyword.status}</span>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
                            {keyword.confidencePercent}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Feature Analysis Summary - ADD KEY PROP */}
                {evaluation.missingFeatureAnalysis && (
                  <div className="card p-6" key={`features-${evaluationKey}`}>
                    <h3 className="text-lg font-semibold mb-4">Feature Analysis</h3>
                    
                    {evaluation.missingFeatureAnalysis.present_features?.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-sm font-semibold text-green-700 mb-2">
                          ✅ Present Features ({evaluation.missingFeatureAnalysis.present_features.length})
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {evaluation.missingFeatureAnalysis.present_features.map((f, i) => (
                            <span key={i} className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                              {f.keyword} ({f.confidence?.toFixed(1)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {evaluation.missingFeatureAnalysis.weak_features?.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-sm font-semibold text-yellow-700 mb-2">
                          ⚠️ Weak Features ({evaluation.missingFeatureAnalysis.weak_features.length})
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {evaluation.missingFeatureAnalysis.weak_features.map((f, i) => (
                            <span key={i} className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs">
                              {f.keyword} ({f.confidence?.toFixed(1)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {evaluation.missingFeatureAnalysis.missing_features?.length > 0 && (
                      <div>
                        <h4 className="text-sm font-semibold text-red-700 mb-2">
                          ❌ Missing Features ({evaluation.missingFeatureAnalysis.missing_features.length})
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {evaluation.missingFeatureAnalysis.missing_features.map((f, i) => (
                            <span key={i} className="px-2 py-1 bg-red-100 text-red-800 rounded text-xs">
                              {f.keyword} ({f.confidence?.toFixed(1)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="card p-6 text-center text-gray-500">
                <ChartBarIcon className="w-12 h-12 mx-auto mb-4 text-gray-400 dark:text-gray-200" />
                <p className="dark:text-white">
                  {isAutoMode 
                    ? isAutoEvaluating
                      ? `Auto-evaluating Image ${currentImageIndex + 1}...`
                      : 'Click "Evaluate Image" or "Auto-Evaluate All" to analyze images'
                    : 'Upload an image and enter a prompt to see evaluation results'
                  }
                </p>
                <p className="text-xs mt-2 dark:text-gray-200">Using Module-2 CLIP-based evaluation</p>
              </div>
            )}
          </div>
        </div>

        {/* NEW: Manual batch indicator (non-auto mode) */}
        {!isAutoMode && manualUploadedImages.length > 0 && (
          <div className="text-center mb-4">
            <div className="inline-flex items-center px-4 py-2 bg-gray-100 text-gray-800 rounded-full text-sm">
              <span className="dark:text-white">Manual Batch: Image {manualIndex + 1} of {manualUploadedImages.length}</span>
            </div>
          </div>
        )}

        {/* Selection toolbars/grids (post-evaluation, manual, etc.) */}
        {(selectingForWatermark || selectingForDownload) && (
          <div className="mt-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-gray-900">
                {selectingForWatermark ? 'Select Images for Watermarking' : 'Select Images to Download'}
              </h3>
              <div className="flex items-center space-x-3">
                {/* UPDATED: show Select All for either mode */}
                {selectingForDownload && (
                  <button
                    className="px-4 py-2 bg-gray-100 text-gray-800 rounded hover:bg-gray-200 transition"
                    onClick={selectAllForDownload}
                  >
                    Select All
                  </button>
                )}
                {/* NEW: Select All in watermark selection mode */}
                {selectingForWatermark && (
                  <button
                    className="px-4 py-2 bg-gray-100 text-gray-800 rounded hover:bg-gray-200 transition"
                    onClick={selectAllForWatermark}
                  >
                    Select All
                  </button>
                )}
                {selectingForWatermark ? (
                  <>
                    <button
                      className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 transition"
                      onClick={sendSelectedToWatermark}
                    >
                      Send to Watermark ({selectedForWatermark.size})
                    </button>
                    <button
                      className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300 transition"
                      onClick={cancelWatermarkSelection}
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      className="px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 transition"
                      onClick={downloadSelectedImages}
                    >
                      Download {selectedForDownload.size} selected image{selectedForDownload.size === 1 ? '' : 's'}
                    </button>
                    <button
                      className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300 transition"
                      onClick={cancelDownloadSelection}
                    >
                      Cancel
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {autoEvaluationImages.map(img => {
                const isSelected = selectingForWatermark
                  ? selectedForWatermark.has(img.id)
                  : selectedForDownload.has(img.id);
                return (
                  <div key={img.id} className="card p-4 relative">
                    <img
                      src={img.url}
                      alt={`generated-${img.id}`}
                      className={`w-full aspect-square object-cover rounded-lg ${isSelected ? 'ring-4 ring-primary-500' : ''}`}
                    />
                    <button
                      onClick={() => (selectingForWatermark ? toggleSelectForWatermark(img.id) : toggleSelectForDownload(img.id))}
                      className="absolute top-3 left-3 w-6 h-6 rounded bg-white border-2 border-gray-300 flex items-center justify-center"
                      title="Select image"
                    >
                      {isSelected && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-primary-600" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-7.25 7.25a1 1 0 01-1.414 0l-3-3a1 1 0 111.414-1.414l2.293 2.293 6.543-6.543a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* NEW: Manual (non-auto) selection toolbar and grid */}
        {(!isAutoMode) && selectingManualForWatermark && (
          <div className="mt-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-gray-900">Select Images for Watermarking</h3>
              <div className="flex items-center space-x-3">
                <button
                  className="px-4 py-2 bg-gray-100 text-gray-800 rounded hover:bg-gray-200 transition"
                  onClick={selectAllManualForWatermark}
                >
                  Select All
                </button>
                <button
                  className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 transition"
                  onClick={sendSelectedManualToWatermark}
                >
                  Send to Watermark ({selectedManualForWatermark.size})
                </button>
                <button
                  className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300 transition"
                  onClick={cancelManualWatermarkSelection}
                >
                  Cancel
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {manualSelectionList.map(img => {
                const isSelected = selectedManualForWatermark.has(img.id);
                return (
                  <div key={img.id} className="card p-4 relative">
                    <img
                      src={img.preview || img.url}
                      alt={img.name || `uploaded-${img.id}`}
                      className={`w-full aspect-square object-cover rounded-lg ${isSelected ? 'ring-4 ring-primary-500' : ''}`}
                    />
                    <button
                      onClick={() => toggleSelectManualForWatermark(img.id)}
                      className="absolute top-3 left-3 w-6 h-6 rounded bg-white border-2 border-gray-300 flex items-center justify-center"
                      title="Select image"
                    >
                      {isSelected && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-primary-600" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-7.25 7.25a1 1 0 01-1.414 0l-3-3a1 1 0 111.414-1.414l2.293 2.293 6.543-6.543a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* NEW: Post-evaluation action dialog */}
      {showPostEvalDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 animate-fade-in">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Post-Evaluation Actions</h3>
              <p className="text-sm text-gray-600 mt-1">
                Evaluations are complete. What would you like to do next?
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                onClick={startPostEvalDownloadSelection}
                className="w-full px-4 py-3 border border-gray-300 bg-white text-gray-800 rounded-lg hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm md:text-base text-left whitespace-normal break-words leading-snug"
              >
                No, Directly select and download generated images
              </button>
              <button
                onClick={startPostEvalWatermarkSelection}
                className="w-full px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm md:text-base text-left whitespace-normal break-words leading-snug"
              >
                Select Images for watermarking
              </button>
            </div>
          </div>
        </div>
      )}

      {/* NEW: Manual post-evaluation dialog (non-auto mode) */}
      {showManualPostEvalDialog && !isAutoMode && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 animate-fade-in">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Do you want to Watermark your analyzed image?</h3>
              <p className="text-sm text-gray-600 mt-1">
                You can select one or more of your uploaded images to watermark.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                onClick={() => setShowManualPostEvalDialog(false)}
                className="w-full px-4 py-3 border border-gray-300 bg-white text-gray-800 rounded-lg hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm md:text-base text-left whitespace-normal break-words leading-snug"
              >
                No
              </button>
              <button
                onClick={startManualWatermarkSelection}
                className="w-full px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm md:text-base text-left whitespace-normal break-words leading-snug"
              >
                Yes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageEvaluator;

