import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from './AuthContext';

interface QueryHistoryItem {
  id: number;
  text: string;
  model_type: string;
  prediction: string;
  confidence: number;
  language: string;
  processing_time: number;
  created_at: string;
}

interface QueryHistoryResponse {
  total_count: number;
  items: QueryHistoryItem[];
  page: number;
  page_size: number;
  total_pages: number;
}

interface QueryHistoryProps {
  onClose: () => void;
}

export const QueryHistory: React.FC<QueryHistoryProps> = ({ onClose }) => {
  const [history, setHistory] = useState<QueryHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [deletingItems, setDeletingItems] = useState<Set<number>>(new Set());
  const [deletingAll, setDeletingAll] = useState(false);
  const [showDeleteAllConfirm, setShowDeleteAllConfirm] = useState(false);
  const { user } = useAuth();

  const API_BASE_URL = 'http://localhost:8000';

  const fetchHistory = async (page: number = 1) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/history`, {
        params: { page, page_size: 10 }
      });
      setHistory(response.data);
      setError('');
    } catch (err: any) {
      console.error('Failed to fetch history:', err);
      setError('Failed to load query history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (user) {
      fetchHistory(currentPage);
    }
  }, [user, currentPage]);

  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
  };

  const deleteHistoryItem = async (itemId: number) => {
    try {
      setDeletingItems(prev => new Set(prev).add(itemId));

      await axios.delete(`${API_BASE_URL}/history/${itemId}`);

      // Refresh history after deletion
      await fetchHistory(currentPage);

    } catch (err: any) {
      console.error('Failed to delete history item:', err);
      setError('Failed to delete history item');
    } finally {
      setDeletingItems(prev => {
        const newSet = new Set(prev);
        newSet.delete(itemId);
        return newSet;
      });
    }
  };

  const deleteAllHistory = async () => {
    try {
      setDeletingAll(true);

      await axios.delete(`${API_BASE_URL}/history`);

      // Refresh history after deletion
      await fetchHistory(1);
      setCurrentPage(1);
      setShowDeleteAllConfirm(false);

    } catch (err: any) {
      console.error('Failed to delete all history:', err);
      setError('Failed to delete all history');
    } finally {
      setDeletingAll(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4CAF50'; // Green
    if (confidence >= 0.6) return '#FF9800'; // Orange
    return '#F44336'; // Red
  };

  const getPredictionEmoji = (modelType: string, prediction: string) => {
    if (modelType === 'sentiment') {
      if (prediction.toLowerCase().includes('positive')) return '😊';
      if (prediction.toLowerCase().includes('negative')) return '😞';
      return '😐';
    }
    if (modelType === 'spam') {
      return prediction.toLowerCase().includes('spam') ? '🚫' : '✅';
    }
    return '📝';
  };

  if (!user) {
    return (
      <div className="history-modal-overlay" onClick={onClose}>
        <div className="history-modal" onClick={(e) => e.stopPropagation()}>
          <div className="history-modal-header">
            <h2>Query History</h2>
            <button className="close-btn" onClick={onClose}>×</button>
          </div>
          <div className="history-content">
            <p>Please log in to view your query history.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="history-modal-overlay" onClick={onClose}>
      <div className="history-modal" onClick={(e) => e.stopPropagation()}>
        <div className="history-modal-header">
          <h2>Query History</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="history-content">
          {loading ? (
            <div className="loading">Loading history...</div>
          ) : error ? (
            <div className="error">{error}</div>
          ) : !history || history.items.length === 0 ? (
            <div className="no-history">
              <p>No query history found. Start classifying some text to see your history here!</p>
            </div>
          ) : (
            <>
              <div className="history-stats">
                <div className="history-stats-info">
                  <p>Total queries: {history.total_count}</p>
                </div>
                <div className="history-actions">
                  <button
                    className="btn-delete-all"
                    onClick={() => setShowDeleteAllConfirm(true)}
                    disabled={deletingAll || history.total_count === 0}
                  >
                    {deletingAll ? '🗑️ Deleting...' : '🗑️ Delete All'}
                  </button>
                </div>
              </div>
              
              <div className="history-list">
                {history.items.map((item) => (
                  <div key={item.id} className="history-item">
                    <div className="history-item-header">
                      <div className="history-item-meta">
                        <span className="model-type">{item.model_type}</span>
                        <span className="timestamp">{formatDate(item.created_at)}</span>
                      </div>
                      <button
                        className="btn-delete-item"
                        onClick={() => deleteHistoryItem(item.id)}
                        disabled={deletingItems.has(item.id)}
                        title="Delete this query"
                      >
                        {deletingItems.has(item.id) ? '⏳' : '🗑️'}
                      </button>
                    </div>
                    
                    <div className="history-item-text">
                      "{item.text.length > 100 ? item.text.substring(0, 100) + '...' : item.text}"
                    </div>
                    
                    <div className="history-item-result">
                      <span className="prediction">
                        {getPredictionEmoji(item.model_type, item.prediction)} {item.prediction}
                      </span>
                      <span 
                        className="confidence"
                        style={{ color: getConfidenceColor(item.confidence) }}
                      >
                        {(item.confidence * 100).toFixed(1)}%
                      </span>
                      <span className="language">🌐 {item.language}</span>
                      <span className="processing-time">⏱️ {item.processing_time.toFixed(2)}s</span>
                    </div>
                  </div>
                ))}
              </div>
              
              {history.total_pages > 1 && (
                <div className="pagination">
                  <button
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="pagination-btn"
                  >
                    Previous
                  </button>
                  
                  <span className="pagination-info">
                    Page {currentPage} of {history.total_pages}
                  </span>
                  
                  <button
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === history.total_pages}
                    className="pagination-btn"
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        {/* Delete All Confirmation Dialog */}
        {showDeleteAllConfirm && (
          <div className="delete-confirm-overlay">
            <div className="delete-confirm-dialog">
              <h3>⚠️ Confirm Delete All</h3>
              <p>
                Are you sure you want to delete all {history?.total_count || 0} query history items?
                This action cannot be undone.
              </p>
              <div className="delete-confirm-actions">
                <button
                  className="btn-cancel"
                  onClick={() => setShowDeleteAllConfirm(false)}
                  disabled={deletingAll}
                >
                  Cancel
                </button>
                <button
                  className="btn-confirm-delete"
                  onClick={deleteAllHistory}
                  disabled={deletingAll}
                >
                  {deletingAll ? 'Deleting...' : 'Delete All'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
