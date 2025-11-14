import React, { useState } from 'react';


const ExtractWatermarkPopup = ({ isOpen, onClose }) => {
  const [image, setImage] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [text, setText] = useState('');
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please upload an image.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    formData.append('image', image);
    formData.append('text', text);
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
        <h2 style={{fontSize: 22, fontWeight: 700, marginBottom: 18, color: '#a5b4fc'}}>Extract Watermark</h2>
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
          <div style={{marginBottom: 18}}>
            <label style={labelStyle}>Watermark Text (optional):</label>
            <input type="text" value={text} onChange={handleTextChange} placeholder="Enter watermark text" style={inputStyle} />
          </div>
          <button type="submit" style={submitBtnStyle} disabled={loading}>{loading ? 'Extracting...' : 'Extract Watermark'}</button>
        </form>
        {error && <div style={errorMsgStyle}>{error}</div>}
        {result && (
          <div style={resultSectionStyle}>
            <h3 style={{fontWeight: 600, fontSize: 17, marginBottom: 8, color: '#a5b4fc'}}>Extraction Result</h3>
            <pre style={{whiteSpace: 'pre-wrap', color: '#e0e7ef', margin: 0}}>{result.output || JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExtractWatermarkPopup;
