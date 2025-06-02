import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface ClassificationResult {
  text: string;
  model_type: string;
  prediction: string;
  confidence: number;
  all_scores: { [key: string]: number };
  temperature: number;
  language: string;
  processing_time: number;
  timestamp: string;
}

interface ApiStatus {
  status: string;
  services: {
    text_classifier: boolean;
    language_detector: boolean;
    database: boolean;
  };
}

interface CSVResultItem {
  row_index: number;
  text: string;
  prediction: string;
  confidence: number;
  language: string;
  processing_time: number;
  error?: string;
}

interface CSVBatchResponse {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  model_type: string;
  total_rows: number;
  processed_rows: number;
  batch_size: number;
  progress_percentage: number;
  results: CSVResultItem[];
  errors: string[];
  started_at: string;
  completed_at?: string;
  processing_time?: number;
}

interface BatchProcessingStatus {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  progress_percentage: number;
  processed_rows: number;
  total_rows: number;
  estimated_time_remaining?: number;
  current_batch: number;
  total_batches: number;
}

const App: React.FC = () => {
  // Single text classification state
  const [text, setText] = useState('');
  const [modelType, setModelType] = useState<'sentiment' | 'spam' | 'topic'>('sentiment');
  const [temperature, setTemperature] = useState(1.0);
  const [temperatureInput, setTemperatureInput] = useState('1.00');
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null);

  // Chart modal state
  const [showChartModal, setShowChartModal] = useState(false);

  // CSV processing state
  const [activeTab, setActiveTab] = useState<'single' | 'csv'>('single');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [batchSize, setBatchSize] = useState(16);
  const [textColumn, setTextColumn] = useState('text');
  const [csvJobId, setCsvJobId] = useState<string | null>(null);
  const [csvStatus, setCsvStatus] = useState<BatchProcessingStatus | null>(null);
  const [csvResults, setCsvResults] = useState<CSVBatchResponse | null>(null);
  const [csvLoading, setCsvLoading] = useState(false);

  // Available batch sizes
  const availableBatchSizes = [1, 4, 8, 16, 64, 128, 256];

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  // Handle ESC key to close modal
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && showChartModal) {
        closeChartModal();
      }
    };

    if (showChartModal) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden'; // Prevent background scrolling
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset'; // Restore scrolling
    };
  }, [showChartModal]);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get('/health');
      setApiStatus(response.data);
    } catch (err) {
      console.error('API health check failed:', err);
      setError('Unable to connect to API. Please make sure the backend is running.');
    }
  };

  const handleClassify = async () => {
    if (!text.trim()) {
      setError('Please enter some text to classify');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('/classify', {
        text: text.trim(),
        model_type: modelType,
        temperature: temperature
      });

      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Classification failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCsvUpload = async () => {
    if (!csvFile) {
      setError('Please select a CSV file');
      return;
    }

    setCsvLoading(true);
    setError('');
    setCsvResults(null);
    setCsvStatus(null);

    try {
      const formData = new FormData();
      formData.append('file', csvFile);
      formData.append('model_type', modelType);
      formData.append('batch_size', batchSize.toString());
      formData.append('text_column', textColumn);

      const response = await axios.post('/classify/csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setCsvJobId(response.data.job_id);
      startPollingStatus(response.data.job_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'CSV upload failed. Please try again.');
      setCsvLoading(false);
    }
  };

  const startPollingStatus = (jobId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const statusResponse = await axios.get(`/classify/csv/status/${jobId}`);
        setCsvStatus(statusResponse.data);

        if (statusResponse.data.status === 'completed' || statusResponse.data.status === 'failed') {
          clearInterval(pollInterval);
          setCsvLoading(false);

          if (statusResponse.data.status === 'completed') {
            // Fetch complete results
            const resultsResponse = await axios.get(`/classify/csv/results/${jobId}`);
            setCsvResults(resultsResponse.data);
          } else {
            setError('CSV processing failed');
          }
        }
      } catch (err: any) {
        clearInterval(pollInterval);
        setCsvLoading(false);
        setError('Failed to get processing status');
      }
    }, 2000); // Poll every 2 seconds
  };

  const downloadResults = () => {
    if (!csvResults) return;

    const csvContent = [
      ['Row Index', 'Text', 'Prediction', 'Confidence', 'Language', 'Processing Time (ms)', 'Error'],
      ...csvResults.results.map(result => [
        result.row_index,
        `"${result.text.replace(/"/g, '""')}"`, // Escape quotes in CSV
        result.prediction,
        (result.confidence * 100).toFixed(2) + '%',
        result.language,
        (result.processing_time * 1000).toFixed(0),
        result.error || ''
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `classification_results_${csvResults.job_id}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#28a745';
    if (confidence >= 0.6) return '#ffc107';
    return '#dc3545';
  };

  const getPredictionEmoji = (prediction: string, modelType: string) => {
    if (modelType === 'sentiment') {
      switch (prediction.toLowerCase()) {
        case 'positive': return '😊';
        case 'negative': return '😞';
        case 'neutral': return '😐';
        default: return '🤔';
      }
    } else if (modelType === 'spam') {
      return prediction.toLowerCase() === 'spam' ? '🚫' : '✅';
    } else {
      return '📝';
    }
  };

  const createChartData = (scores: { [key: string]: number }) => {
    const labels = Object.keys(scores);
    const data = Object.values(scores);
    const backgroundColors = labels.map((_, index) => {
      const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'];
      return colors[index % colors.length];
    });

    return {
      labels: labels.map(label => label.charAt(0).toUpperCase() + label.slice(1)),
      datasets: [
        {
          label: 'Confidence Score',
          data: data.map(score => (score * 100).toFixed(1)),
          backgroundColor: backgroundColors,
          borderColor: backgroundColors.map(color => color + '80'),
          borderWidth: 1,
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Classification Scores (%)',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value: any) {
            return value + '%';
          }
        }
      },
    },
  };

  // Temperature validation and normalization functions
  const normalizeTemperature = (value: number): number => {
    return Math.max(0.5, Math.min(2.0, value));
  };

  const handleTemperatureSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTemp = parseFloat(e.target.value);
    setTemperature(newTemp);
    setTemperatureInput(newTemp.toFixed(2));
  };

  const handleTemperatureInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setTemperatureInput(value);
    
    // Try to parse the input value
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      const normalizedTemp = normalizeTemperature(numValue);
      setTemperature(normalizedTemp);
    }
  };

  const handleTemperatureInputBlur = () => {
    const numValue = parseFloat(temperatureInput);
    if (isNaN(numValue)) {
      // If invalid input, reset to current temperature
      setTemperatureInput(temperature.toFixed(2));
    } else {
      // Normalize and update both values
      const normalizedTemp = normalizeTemperature(numValue);
      setTemperature(normalizedTemp);
      setTemperatureInput(normalizedTemp.toFixed(2));
    }
  };

  const handleTemperatureInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleTemperatureInputBlur();
    }
  };

  // Chart modal functions
  const openChartModal = () => {
    setShowChartModal(true);
  };

  const closeChartModal = () => {
    setShowChartModal(false);
  };

  const handleModalBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      closeChartModal();
    }
  };

  // Enhanced chart options for modal view
  const modalChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          font: {
            size: 14
          }
        }
      },
      title: {
        display: true,
        text: `Classification Scores (%) - Temperature: ${temperature.toFixed(2)}`,
        font: {
          size: 18,
          weight: 'bold' as const
        }
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value: any) {
            return value + '%';
          },
          font: {
            size: 12
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      x: {
        ticks: {
          font: {
            size: 12
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🤖 Text Classification Demo</h1>
          <p>Analyze text sentiment, detect spam, and classify topics with AI</p>
          
          {apiStatus && (
            <div className={`status ${apiStatus.status === 'healthy' ? 'success' : 'error'}`}>
              API Status: {apiStatus.status} 
              {apiStatus.status === 'healthy' && ' ✅'}
            </div>
          )}
        </header>

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button
            className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
            onClick={() => setActiveTab('single')}
          >
            Single Text
          </button>
          <button
            className={`tab-btn ${activeTab === 'csv' ? 'active' : ''}`}
            onClick={() => setActiveTab('csv')}
          >
            Batch CSV
          </button>
        </div>

        {/* Model Selection and Temperature Control */}
        <div className="card">
          <div className="model-config-container">
            <div className="model-selection-section">
              <div className="input-group">
                <label>Select Classification Model:</label>
                <div className="model-selection">
                  <div className={`model-option ${modelType === 'sentiment' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      id="sentiment"
                      name="modelType"
                      value="sentiment"
                      checked={modelType === 'sentiment'}
                      onChange={(e) => setModelType(e.target.value as 'sentiment' | 'spam' | 'topic')}
                    />
                    <label htmlFor="sentiment" className="model-label">
                      <span className="model-emoji">😊</span>
                      <div className="model-text">
                        <span className="model-name">Sentiment</span>
                        <span className="model-description">Positive/Negative/Neutral</span>
                      </div>
                    </label>
                  </div>

                  <div className={`model-option ${modelType === 'spam' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      id="spam"
                      name="modelType"
                      value="spam"
                      checked={modelType === 'spam'}
                      onChange={(e) => setModelType(e.target.value as 'sentiment' | 'spam' | 'topic')}
                    />
                    <label htmlFor="spam" className="model-label">
                      <span className="model-emoji">🚫</span>
                      <div className="model-text">
                        <span className="model-name">Spam Detection</span>
                        <span className="model-description">Spam/Not Spam</span>
                      </div>
                    </label>
                  </div>

                  <div className={`model-option ${modelType === 'topic' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      id="topic"
                      name="modelType"
                      value="topic"
                      checked={modelType === 'topic'}
                      onChange={(e) => setModelType(e.target.value as 'sentiment' | 'spam' | 'topic')}
                    />
                    <label htmlFor="topic" className="model-label">
                      <span className="model-emoji">🏷️</span>
                      <div className="model-text">
                        <span className="model-name">Topic</span>
                        <span className="model-description">Multi-category</span>
                      </div>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <div className="temperature-control">
              <div className="input-group">
                <label htmlFor="temperature-slider">
                  🌡️ Temperature Control
                </label>
                <div className="temperature-info">
                  <small>Controls prediction confidence: Lower = more confident, Higher = less confident</small>
                </div>
                
                <div className="temperature-input-section">
                  <div className="temperature-value-input">
                    <label htmlFor="temperature-input">Value:</label>
                    <input
                      type="number"
                      id="temperature-input"
                      min="0.5"
                      max="2.0"
                      step="0.01"
                      value={temperatureInput}
                      onChange={handleTemperatureInputChange}
                      onBlur={handleTemperatureInputBlur}
                      onKeyPress={handleTemperatureInputKeyPress}
                      className="temperature-number-input"
                      placeholder="0.50 - 2.00"
                    />
                  </div>
                  <div className="temperature-range">
                    <span className="range-label">Range: 0.50 - 2.00</span>
                  </div>
                </div>

                <div className="slider-container">
                  <span className="slider-label">0.5</span>
                  <input
                    type="range"
                    id="temperature-slider"
                    min="0.5"
                    max="2.0"
                    step="0.05"
                    value={temperature}
                    onChange={handleTemperatureSliderChange}
                    className="temperature-slider"
                  />
                  <span className="slider-label">2.0</span>
                </div>
                
                <div className={`temperature-description ${
                  temperature < 0.8 ? 'high-confidence' : 
                  temperature <= 1.2 ? 'balanced' : 'exploratory'
                }`}>
                  {temperature < 0.8 && "🔥 High Confidence Mode"}
                  {temperature >= 0.8 && temperature <= 1.2 && "⚖️ Balanced Mode"}
                  {temperature > 1.2 && "🌈 Exploratory Mode"}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Single Text Tab */}
        {activeTab === 'single' && (
          <div className="card">
            <div className="input-group">
              <label htmlFor="text-input">Enter text to classify:</label>
              <textarea
                id="text-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type your text here... (e.g., 'I love this product!' or 'Free money! Click now!')"
                rows={4}
                maxLength={10000}
              />
              <small>{text.length}/10000 characters</small>
            </div>

            <button
              className="btn"
              onClick={handleClassify}
              disabled={loading || !text.trim()}
            >
              {loading ? 'Analyzing...' : 'Classify Text'}
            </button>

            {error && (
              <div className="error">
                ❌ {error}
              </div>
            )}

            {loading && (
              <div className="loading">
                <div className="spinner"></div>
                <span style={{ marginLeft: '10px' }}>Processing your text...</span>
              </div>
            )}

            {result && (
              <div className="result-card">
                <h3>📊 Classification Result</h3>

                <div className="result-grid">
                  <div className="result-info">
                    <div className="result-item">
                      <strong>Text:</strong> "{result.text}"
                    </div>

                    <div className="result-item">
                      <strong>Model:</strong> {result.model_type}
                    </div>

                    <div className="result-item">
                      <strong>Temperature:</strong> {result.temperature.toFixed(2)}
                    </div>

                    <div className="result-item">
                      <strong>Prediction:</strong>
                      <span className="prediction">
                        {getPredictionEmoji(result.prediction, result.model_type)} {result.prediction}
                      </span>
                    </div>

                    <div className="result-item">
                      <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{
                            width: `${result.confidence * 100}%`,
                            backgroundColor: getConfidenceColor(result.confidence)
                          }}
                        ></div>
                      </div>
                    </div>

                    <div className="result-item">
                      <strong>Language:</strong> {result.language.toUpperCase()}
                    </div>

                    <div className="result-item">
                      <strong>Processing Time:</strong> {(result.processing_time * 1000).toFixed(0)}ms
                    </div>
                  </div>

                  {result.all_scores && Object.keys(result.all_scores).length > 0 && (
                    <div className="result-chart">
                      <h4>📈 All Label Scores</h4>
                      <div className="chart-container" onClick={openChartModal}>
                        <Bar data={createChartData(result.all_scores)} options={chartOptions} />
                        <div className="chart-overlay">
                          <div className="expand-icon">🔍</div>
                          <div className="expand-text">Click to enlarge</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* CSV Batch Tab */}
        {activeTab === 'csv' && (
          <div className="card">
            <h3>📄 CSV Batch Processing</h3>

            <div className="input-group">
              <label htmlFor="csv-file">Select CSV File:</label>
              <input
                type="file"
                id="csv-file"
                accept=".csv"
                onChange={(e) => setCsvFile(e.target.files?.[0] || null)}
                className="file-input"
              />
              {csvFile && (
                <small>Selected: {csvFile.name} ({(csvFile.size / 1024).toFixed(1)} KB)</small>
              )}
            </div>

            <div className="csv-config">
              <div className="input-group">
                <label htmlFor="text-column">Text Column Name:</label>
                <input
                  type="text"
                  id="text-column"
                  value={textColumn}
                  onChange={(e) => setTextColumn(e.target.value)}
                  placeholder="text"
                  className="text-input"
                />
                <small>Name of the column containing text to classify</small>
              </div>

              <div className="input-group">
                <label htmlFor="batch-size">Batch Size:</label>
                <div className="batch-size-slider">
                  <input
                    type="range"
                    id="batch-size"
                    min={0}
                    max={availableBatchSizes.length - 1}
                    value={availableBatchSizes.indexOf(batchSize)}
                    onChange={(e) => setBatchSize(availableBatchSizes[parseInt(e.target.value)])}
                    className="slider"
                  />
                  <div className="slider-labels">
                    {availableBatchSizes.map((size, index) => (
                      <span 
                        key={size} 
                        className={`slider-label ${availableBatchSizes.indexOf(batchSize) === index ? 'active' : ''}`}
                      >
                        {size}
                      </span>
                    ))}
                  </div>
                  <div className="batch-size-display">
                    <strong>{batchSize}</strong> {batchSize === 1 ? 'text' : 'texts'} per batch
                  </div>
                </div>
                <small>Choose the number of texts to process in each batch</small>
              </div>
            </div>

            <button
              className="btn"
              onClick={handleCsvUpload}
              disabled={csvLoading || !csvFile}
            >
              {csvLoading ? 'Processing...' : 'Upload & Process CSV'}
            </button>

            {error && (
              <div className="error">
                ❌ {error}
              </div>
            )}

            {csvLoading && csvStatus && (
              <div className="progress-section">
                <h4>📊 Processing Progress</h4>
                <div className="progress-info">
                  <div>Status: <strong>{csvStatus.status}</strong></div>
                  <div>Progress: <strong>{csvStatus.progress_percentage.toFixed(1)}%</strong></div>
                  <div>Processed: <strong>{csvStatus.processed_rows}/{csvStatus.total_rows}</strong> rows</div>
                  <div>Current Batch: <strong>{csvStatus.current_batch}/{csvStatus.total_batches}</strong></div>
                  {csvStatus.estimated_time_remaining && (
                    <div>Estimated Time Remaining: <strong>{csvStatus.estimated_time_remaining.toFixed(0)}s</strong></div>
                  )}
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${csvStatus.progress_percentage}%` }}
                  ></div>
                </div>
              </div>
            )}

            {csvResults && csvResults.status === 'completed' && (
              <div className="csv-results">
                <div className="results-header">
                  <h4>✅ Processing Complete</h4>
                  <button className="btn btn-secondary" onClick={downloadResults}>
                    📥 Download Results
                  </button>
                </div>

                <div className="results-summary">
                  <div>Total Processed: <strong>{csvResults.processed_rows}</strong> rows</div>
                  <div>Processing Time: <strong>{csvResults.processing_time?.toFixed(1)}s</strong></div>
                  <div>Batch Size: <strong>{csvResults.batch_size}</strong></div>
                  {csvResults.errors.length > 0 && (
                    <div>Errors: <strong>{csvResults.errors.length}</strong></div>
                  )}
                </div>

                <div className="results-table-container">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>Row</th>
                        <th>Text</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Language</th>
                        <th>Time (ms)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {csvResults.results.slice(0, 10).map((result, index) => (
                        <tr key={index}>
                          <td>{result.row_index}</td>
                          <td className="text-cell" title={result.text}>
                            {result.text.length > 50 ? result.text.substring(0, 50) + '...' : result.text}
                          </td>
                          <td>
                            <span className="prediction">
                              {getPredictionEmoji(result.prediction, csvResults.model_type)} {result.prediction}
                            </span>
                          </td>
                          <td>
                            <span style={{ color: getConfidenceColor(result.confidence) }}>
                              {(result.confidence * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td>{result.language.toUpperCase()}</td>
                          <td>{(result.processing_time * 1000).toFixed(0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {csvResults.results.length > 10 && (
                    <div className="table-note">
                      Showing first 10 results. Download CSV for complete results.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Try These Examples - Only for Single Text Tab */}
        {activeTab === 'single' && (
          <div className="card">
            <h3>🎯 Try These Examples:</h3>
            <div className="examples">
              {modelType === 'sentiment' && (
                <div className="example-group">
                  <h4>Sentiment Analysis 😊:</h4>
                  <button 
                    className="example-btn"
                    onClick={() => setText("I absolutely love this product! It's amazing and works perfectly!")}
                  >
                    Positive Example
                  </button>
                  <button 
                    className="example-btn"
                    onClick={() => setText("This is terrible. I hate it and want my money back.")}
                  >
                    Negative Example
                  </button>
                  <button 
                    className="example-btn"
                    onClick={() => setText("I went to the store to buy some groceries.")}
                  >
                    Neutral Example
                  </button>
                </div>
              )}
              
              {modelType === 'spam' && (
                <div className="example-group">
                  <h4>Spam Detection 🚫:</h4>
                  <button 
                    className="example-btn"
                    onClick={() => setText("FREE MONEY! Click now to win $1000! Limited time offer!")}
                  >
                    Spam Example
                  </button>
                  <button 
                    className="example-btn"
                    onClick={() => setText("Hi, I wanted to follow up on our meeting yesterday about the project timeline.")}
                  >
                    Not Spam Example
                  </button>
                </div>
              )}

              {modelType === 'topic' && (
                <div className="example-group">
                  <h4>Topic Classification 🏷️:</h4>
                  <button 
                    className="example-btn"
                    onClick={() => setText("The new AI programming language makes machine learning development much faster and easier for developers.")}
                  >
                    Technology Example
                  </button>
                  <button 
                    className="example-btn"
                    onClick={() => setText("The basketball team won the championship after an incredible final game with a score of 95-88.")}
                  >
                    Sports Example
                  </button>
                  <button 
                    className="example-btn"
                    onClick={() => setText("The company's quarterly earnings exceeded expectations, driving stock prices up by 15% in early trading.")}
                  >
                    Business Example
                  </button>
                </div>  
              )}
            </div>
          </div>
        )}

        {/* Chart Modal */}
        {showChartModal && result && result.all_scores && (
          <div className="chart-modal-overlay" onClick={handleModalBackdropClick}>
            <div className="chart-modal">
              <div className="chart-modal-header">
                <h3>📊 Detailed Classification Scores</h3>
                <button className="modal-close-btn" onClick={closeChartModal}>
                  ✕
                </button>
              </div>
              <div className="chart-modal-content">
                <div className="modal-chart-info">
                  <div className="modal-info-item">
                    <strong>Model:</strong> {result.model_type}
                  </div>
                  <div className="modal-info-item">
                    <strong>Temperature:</strong> {result.temperature.toFixed(2)}
                  </div>
                  <div className="modal-info-item">
                    <strong>Best Prediction:</strong> 
                    <span className="prediction">
                      {getPredictionEmoji(result.prediction, result.model_type)} {result.prediction}
                    </span>
                  </div>
                </div>
                <div className="modal-chart-container">
                  <Bar data={createChartData(result.all_scores)} options={modalChartOptions} />
                </div>
                <div className="modal-scores-table">
                  <h4>Score Details</h4>
                  <div className="scores-grid">
                    {Object.entries(result.all_scores)
                      .sort(([,a], [,b]) => b - a)
                      .map(([label, score], index) => (
                        <div key={label} className={`score-item ${index === 0 ? 'best-score' : ''}`}>
                          <div className="score-label">
                            {label.charAt(0).toUpperCase() + label.slice(1)}
                            {index === 0 && <span className="best-badge">BEST</span>}
                          </div>
                          <div className="score-value">{(score * 100).toFixed(2)}%</div>
                          <div className="score-bar">
                            <div 
                              className="score-fill" 
                              style={{ 
                                width: `${score * 100}%`,
                                backgroundColor: index === 0 ? '#28a745' : '#6c757d'
                              }}
                            ></div>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        <footer className="footer">
          <p>🚀 Text Classification System Demo - Built with React & FastAPI</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
