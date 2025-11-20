import React, { useState } from 'react';


const ExtractWatermarkPopup = ({ isOpen, onClose }) => {
  const [image, setImage] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [text, setText] = useState('');
  const [watermarkImage, setWatermarkImage] = useState(null);  // NEW
  const [verificationMode, setVerificationMode] = useState('text');  // NEW: 'text' or 'image'
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);


  const handleImageChange = (e) => {
    const file = e.target.files && e.target.files[0];
    setImage(file || null);
    setResult(null);
    setError(null);
  };

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setImage(e.dataTransfer.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  // NEW: Handle watermark image upload
  const handleWatermarkImageChange = (e) => {
    const file = e.target.files && e.target.files[0];
    setWatermarkImage(file || null);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please upload an image.');
      return;
    }
    
    // Validate verification input
    if (verificationMode === 'text' && !text.trim()) {
      // Allow extraction without verification
      console.log('[EXTRACT] No text provided - will extract only');
    } else if (verificationMode === 'image' && !watermarkImage) {
      setError('Please upload the watermark image for verification.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);
    
    const formData = new FormData();
    formData.append('image', image);
    
    // Add verification data based on mode
    if (verificationMode === 'text') {
      formData.append('text', text);
    } else if (verificationMode === 'image' && watermarkImage) {
      formData.append('watermark_image', watermarkImage);
    }
    
    try {
      const response = await fetch('/api/extract-watermark', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setResult(data);
      } else {
        setError(data.error || 'Extraction failed.');
      }
    } catch (err) {
      setError('Server error.');
    }
    setLoading(false);
  };

  if (!isOpen) return null;

  // Styles (dark mode and theme-aware)
  const overlayStyle = {
    position: 'fixed',
    top: 0, left: 0, right: 0, bottom: 0,
    background: 'rgba(0,0,0,0.5)',
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };
  const popupStyle = {
    background: 'var(--popup-bg, #18181b)',
    color: 'var(--popup-fg, #f3f4f6)',
    borderRadius: '0.75rem',
    padding: '1.2rem 0.7rem',
    minWidth: '60vw',
    maxWidth: 290,
    width: '80vw',
    minHeight: 220,
    maxHeight: '90vh',
    boxShadow: '0 2px 16px rgba(0,0,0,0.3)',
    position: 'relative',
    fontFamily: 'inherit',
    border: '1px solid #27272a',
    display: 'flex',
    flexDirection: 'column',
    transition: 'width 0.3s cubic-bezier(.4,2,.6,1), height 0.3s cubic-bezier(.4,2,.6,1)',
    overflow: 'hidden',
    boxSizing: 'border-box',
    '@media (min-width: 600px)': {
      minWidth: 200,
      width: 'auto',
    },
  };
  const closeBtnStyle = {
    position: 'absolute',
    top: 12,
    right: 16,
    background: 'none',
    border: 'none',
    fontSize: '1.5rem',
    color: '#a1a1aa',
    cursor: 'pointer',
    transition: 'color 0.2s',
  };
  const labelStyle = {
    display: 'block',
    marginBottom: 6,
    fontWeight: 500,
    color: '#d4d4d8',
    fontSize: 14,
  };
  const inputStyle = {
    width: '100%',
    padding: '0.5rem',
    borderRadius: 6,
    border: '1px solid #27272a',
    background: '#23232b',
    color: '#f3f4f6',
    fontSize: 15,
    marginBottom: 12,
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border 0.2s',
  };
  const fileInputStyle = {
    ...inputStyle,
    padding: 0,
    background: 'transparent',
    color: '#f3f4f6',
    border: 'none',
    marginBottom: 12,
    display: 'none',
  };
  const dropZoneStyle = {
    border: dragActive ? '2px solid #6366f1' : '2px dashed #52525b',
    background: dragActive ? '#23234b' : '#23232b',
    borderRadius: 8,
    padding: '1.2rem 0.5rem',
    textAlign: 'center',
    color: '#a1a1aa',
    cursor: 'pointer',
    marginBottom: 12,
    transition: 'border 0.2s, background 0.2s',
    fontSize: 15,
    position: 'relative',
    minHeight: 48,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'column',
  };
  const submitBtnStyle = {
    background: 'linear-gradient(90deg,#6366f1,#2563eb)',
    color: '#fff',
    border: 'none',
    padding: '0.6rem 1.5rem',
    borderRadius: 6,
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: 16,
    marginTop: 8,
    width: '100%',
    boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
    opacity: loading ? 0.7 : 1,
    transition: 'opacity 0.2s',
  };
  const errorMsgStyle = {
    color: '#f87171',
    marginTop: 12,
    fontSize: 14,
    fontWeight: 500,
  };
  const resultSectionStyle = {
    marginTop: 24,
    background: '#23232b',
    padding: 16,
    borderRadius: 8,
    fontSize: 15,
    color: '#e0e7ef',
    wordBreak: 'break-all',
    border: '1px solid #27272a',
    maxHeight: '32vh',
    overflowY: 'auto',
    transition: 'max-height 0.3s cubic-bezier(.4,2,.6,1)',
  };

  return (
    <div style={overlayStyle}>
      <div style={popupStyle}>
        <button style={closeBtnStyle} onClick={onClose} title="Close">&times;</button>
        <h2 style={{fontSize: 22, fontWeight: 700, marginBottom: 18, color: '#a5b4fc'}}>Verify Watermark</h2>
        <form onSubmit={handleSubmit} autoComplete="off" style={{flexShrink: 0}}>
          <div style={{marginBottom: 18}}>
            <label style={labelStyle}>Upload Image:</label>
            <div
              style={dropZoneStyle}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => document.getElementById('extract-wm-file-input').click()}
              tabIndex={0}
              role="button"
              aria-label="Upload image"
            >
              <input
                id="extract-wm-file-input"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                style={fileInputStyle}
              />
              {image ? (
                <span style={{color: '#a5b4fc', fontWeight: 500}}>{image.name}</span>
              ) : (
                <span>Drag & drop or click to select image</span>
              )}
            </div>
          </div>
          
          {/* NEW: Verification mode selector */}
          <div style={{marginBottom: 18}}>
            <label style={labelStyle}>Verification Method:</label>
            <div style={{display: 'flex', gap: 16, marginBottom: 12}}>
              <label style={{display: 'flex', alignItems: 'center', cursor: 'pointer'}}>
                <input
                  type="radio"
                  value="text"
                  checked={verificationMode === 'text'}
                  onChange={(e) => setVerificationMode(e.target.value)}
                  style={{marginRight: 6}}
                />
                <span>Text Watermark</span>
              </label>
              <label style={{display: 'flex', alignItems: 'center', cursor: 'pointer'}}>
                <input
                  type="radio"
                  value="image"
                  checked={verificationMode === 'image'}
                  onChange={(e) => setVerificationMode(e.target.value)}
                  style={{marginRight: 6}}
                />
                <span>Image Watermark</span>
              </label>
            </div>
          </div>
          
          {/* Conditional input based on verification mode */}
          {verificationMode === 'text' ? (
            <div style={{marginBottom: 18}}>
              <label style={labelStyle}>Watermark Text :</label>
              <input 
                type="text" 
                value={text} 
                onChange={handleTextChange} 
                placeholder="Enter watermark text" 
                style={inputStyle} 
              />
              <p style={{fontSize: 12, color: '#a1a1aa', marginTop: 4}}>
               
              </p>
            </div>
          ) : (
            <div style={{marginBottom: 18}}>
              <label style={labelStyle}>Watermark Image:</label>
              <input
                type="file"
                accept="image/*"
                onChange={handleWatermarkImageChange}
                style={{...inputStyle, display: 'block', padding: '0.5rem'}}
              />
              {watermarkImage && (
                <p style={{fontSize: 12, color: '#a5b4fc', marginTop: 4}}>
                  Selected: {watermarkImage.name}
                </p>
              )}
            </div>
          )}
          
          <button type="submit" style={submitBtnStyle} disabled={loading}>
            {loading ? 'Verifying...' : 'Extract & Verify Watermark'}
          </button>
        </form>
        
        {error && <div style={errorMsgStyle}>{error}</div>}
        {result && (
          <div style={resultSectionStyle}>
            {/* Check if new format (extraction/verification) or old format */}
            {result.extraction ? (
              <div className="space-y-4">
                {/* EXTRACTION RESULTS */}
                <div>
                  <h3 style={{fontSize: 16, fontWeight: 600, marginBottom: 12, color: '#a5b4fc'}}>
                    📊 Extraction Results
                  </h3>
                  
                  <div style={{fontSize: 14, color: '#d4d4d8'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                      <span>Watermark Size:</span>
                      <strong>{result.extraction.watermark_size}×{result.extraction.watermark_size}</strong>
                    </div>
                    <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                      <span>Redundancy:</span>
                      <strong>{result.extraction.redundancy}</strong>
                    </div>
                    <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                      <span>Pattern Density:</span>
                      <strong>{(result.extraction.density * 100).toFixed(1)}%</strong>
                    </div>
                    
                    {result.extraction.metadata_text && (
                      <div style={{marginTop: 12, padding: 8, background: '#2a2a32', borderRadius: 6}}>
                        <div style={{fontSize: 12, color: '#a1a1aa', marginBottom: 4}}>From PNG metadata:</div>
                        <div style={{fontWeight: 600, color: '#a5b4fc'}}>"{result.extraction.metadata_text}"</div>
                      </div>
                    )}
                  </div>

                  {/* Extracted Pattern Visualization */}
                  {result.extraction.pattern_visualization && (
                    <div style={{marginTop: 16}}>
                      <div style={{fontSize: 12, color: '#a1a1aa', marginBottom: 8}}>Extracted Binary Pattern:</div>
                      <div style={{display: 'flex', gap: 12, alignItems: 'flex-start'}}>
                        <img
                          src={result.extraction.pattern_visualization}
                          alt="Extracted pattern"
                          style={{
                            width: 120,
                            height: 120,
                            border: '2px solid #52525b',
                            borderRadius: 6,
                            imageRendering: 'pixelated'
                          }}
                        />
                        <div style={{flex: 1, fontSize: 12, color: '#a1a1aa'}}>
                          <div>• White = +1 bit</div>
                          <div>• Black = -1 bit</div>
                          <div>• {result.extraction.watermark_size}×{result.extraction.watermark_size} grid</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* VERIFICATION RESULTS */}
                {result.verification.attempted && (
                  <div style={{
                    marginTop: 16,
                    padding: 12,
                    borderRadius: 8,
                    background: result.verification.result === 'verified' 
                      ? '#065f46' 
                      : result.verification.result === 'mismatch'
                      ? '#7f1d1d'
                      : result.verification.result === 'mode_mismatch'
                      ? '#78350f'
                      : '#78350f'
                  }}>
                    <h3 style={{fontSize: 16, fontWeight: 600, marginBottom: 12}}>
                      {result.verification.result === 'verified' ? '✅' : 
                       result.verification.result === 'mismatch' ? '❌' : '⚠️'}
                      {' '}Verification Results
                    </h3>
                    
                    <div style={{fontSize: 14}}>
                      {/* Show claimed text OR image indicator */}
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                        <span>{result.verification.claimed_image ? 'Claimed Watermark:' : 'Claimed Text:'}</span>
                        <strong>
                          {result.verification.claimed_image 
                            ? '🖼️ Image Watermark' 
                            : `"${result.verification.claimed_text}"`}
                        </strong>
                      </div>
                      
                      {/* Show NCC score unless mode mismatch */}
                      {result.verification.result !== 'mode_mismatch' && (
                        <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                          <span>NCC Score:</span>
                          <strong style={{
                            color: result.verification.ncc_score > 0.6 ? '#4ade80' :
                                   result.verification.ncc_score > 0.3 ? '#fbbf24' : '#f87171'
                          }}>
                            {result.verification.ncc_score.toFixed(4)}
                          </strong>
                        </div>
                      )}
                      
                      {/* Success message */}
                      {result.verification.verified_text && (
                        <div style={{marginTop: 12, padding: 10, background: '#064e3b', borderRadius: 6, fontWeight: 600}}>
                          ✓ Verified: {result.verification.claimed_image ? '🖼️ Image Watermark Matches!' : `"${result.verification.verified_text}"`}
                        </div>
                      )}
                      
                      {/* Mismatch message */}
                      {result.verification.result === 'mismatch' && (
                        <p style={{marginTop: 12, fontSize: 13, color: '#fca5a5'}}>
                          {result.verification.claimed_image 
                            ? 'The uploaded watermark image does not match the embedded watermark.' 
                            : 'The claimed text does not match the embedded watermark.'}
                        </p>
                      )}
                      
                      {/* Mode mismatch warning */}
                      {result.verification.result === 'mode_mismatch' && result.verification.message && (
                        <p style={{marginTop: 12, fontSize: 13, color: '#fbbf24'}}>
                          ⚠️ {result.verification.message}
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {!result.verification.attempted && (
                  <div style={{marginTop: 16, padding: 12, background: '#2a2a32', borderRadius: 8, fontSize: 14, color: '#a1a1aa'}}>
                    💡 {verificationMode === 'text' 
                      ? 'Enter watermark text above and verify again.' 
                      : 'Upload the watermark image above and verify again.'}
                  </div>
                )}
              </div>
            ) : (
              /* OLD FORMAT FALLBACK */
              <div>
                <h3 style={{fontSize: 16, fontWeight: 600, marginBottom: 12, color: '#a5b4fc'}}>Results</h3>
                <pre style={{fontSize: 13, whiteSpace: 'pre-wrap', color: '#d4d4d8'}}>{JSON.stringify(result, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ExtractWatermarkPopup;
