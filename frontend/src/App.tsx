import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

interface ClassificationResult {
  text: string;
  model_type: string;
  prediction: string;
  confidence: number;
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
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null);

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
        model_type: modelType
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

        {/* Model Selection (shared between tabs) */}
        <div className="card">
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
                    <span className="model-description">Positive/Negative</span>
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

                <div className="result-item">
                  <strong>Text:</strong> "{result.text}"
                </div>

                <div className="result-item">
                  <strong>Model:</strong> {result.model_type}
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
                <button 
                  className="example-btn"
                  onClick={() => setText("New research shows that regular exercise and a balanced diet can significantly reduce the risk of heart disease.")}
                >
                  Health Example
                </button>
                <button 
                  className="example-btn"
                  onClick={() => setText("The university announced new scholarship programs for students pursuing computer science and engineering degrees.")}
                >
                  Education Example
                </button>
                <button 
                  className="example-btn"
                  onClick={() => setText("The latest blockbuster movie starring famous actors broke box office records on its opening weekend.")}
                >
                  Entertainment Example
                </button>
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
