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

const App: React.FC = () => {
  const [text, setText] = useState('');
  const [modelType, setModelType] = useState<'sentiment' | 'spam' | 'topic'>('sentiment');
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null);

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
            </div>
          </div>
        </div>

        <footer className="footer">
          <p>🚀 Text Classification System Demo - Built with React & FastAPI</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
