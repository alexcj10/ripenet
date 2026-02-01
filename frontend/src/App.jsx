import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Sparkles, Timer, ArrowRight, RotateCcw, Check, AlertCircle, Github } from 'lucide-react';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      setFile(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setResult(response.data);
    } catch (err) {
      setError('Could not analyze. Please check the backend.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const getRipenessStyle = (ripeness) => {
    if (ripeness === 'Fresh') return { bg: '#dcfce7', color: '#16a34a', icon: '🍃' };
    if (ripeness === 'Unripe') return { bg: '#fef3c7', color: '#d97706', icon: '🌱' };
    return { bg: '#fee2e2', color: '#dc2626', icon: '🍂' };
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <img src="/apple.svg" alt="RipeNet" className="logo-icon" />
          <span className="logo-text">RipeNet</span>
        </div>
        <div className="header-right">
          <div className="badge">AI Powered</div>
          <a href="https://github.com/alexcj10/ripenet" target="_blank" rel="noopener noreferrer" className="github-link">
            <Github size={22} />
          </a>
        </div>
      </header>

      {/* Hero */}
      <section className="hero">
        <h1>Know Your Fruit's <span className="highlight">Freshness</span></h1>
        <p>Upload a photo and get instant AI analysis of ripeness and shelf-life.</p>
      </section>

      {/* Main Content */}
      <main className="main">
        <AnimatePresence mode="wait">
          {!result ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="upload-section"
            >
              {!preview ? (
                <label
                  className={`dropzone ${dragOver ? 'drag-over' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                >
                  <input type="file" hidden accept="image/*" onChange={handleFileChange} />
                  <div className="dropzone-content">
                    <div className="upload-icon-wrap">
                      <Upload size={32} />
                    </div>
                    <p className="dropzone-title">Drop your fruit image here</p>
                    <p className="dropzone-hint">or click to browse</p>
                    <div className="format-tags">
                      <span>JPG</span>
                      <span>PNG</span>
                      <span>WEBP</span>
                    </div>
                  </div>
                </label>
              ) : (
                <div className="preview-section">
                  <div className="preview-wrapper">
                    <img src={preview} alt="Preview" className="preview-image" />
                    <button className="change-btn" onClick={reset}>
                      <RotateCcw size={16} /> Change
                    </button>
                  </div>

                  <button
                    className={`analyze-btn ${loading ? 'loading' : ''}`}
                    onClick={handleUpload}
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Sparkles size={20} />
                        Analyze Freshness
                        <ArrowRight size={18} />
                      </>
                    )}
                  </button>

                  {error && (
                    <div className="error-box">
                      <AlertCircle size={18} />
                      {error}
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="result-section"
            >
              <div className="result-header">
                <Check size={24} className="check-icon" />
                <span>Analysis Complete</span>
              </div>

              <div className="result-content">
                <div className="fruit-info">
                  <div className="fruit-image-wrap">
                    <img src={preview} alt="Analyzed fruit" className="fruit-image" />
                  </div>
                  <div className="fruit-name">{result.fruit}</div>
                </div>

                <div className="stats-grid">
                  <div
                    className="stat-card"
                    style={{ background: getRipenessStyle(result.ripeness).bg }}
                  >
                    <span className="stat-emoji">{getRipenessStyle(result.ripeness).icon}</span>
                    <span className="stat-label">Ripeness</span>
                    <span
                      className="stat-value"
                      style={{ color: getRipenessStyle(result.ripeness).color }}
                    >
                      {result.ripeness}
                    </span>
                  </div>

                  <div className="stat-card shelf-life">
                    <Timer size={28} className="timer-icon" />
                    <span className="stat-label">Shelf Life</span>
                    <span className="stat-value">{result.shelf_life_days} days</span>
                  </div>
                </div>

                <div className="report-box">
                  <p>"{result.report}"</p>
                </div>

                <button className="new-btn" onClick={reset}>
                  <RotateCcw size={18} />
                  Analyze Another Fruit
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Powered by Deep Learning 🧠</p>
      </footer>
    </div>
  );
}

export default App;
