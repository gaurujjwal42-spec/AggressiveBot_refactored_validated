import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import TradeDetailsModal from './TradeDetailsModal';

const API_BASE_URL = 'http://127.0.0.1:5000/api';
const formatCurrency = (value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value || 0);
const formatDate = (isoString) => isoString ? new Date(isoString).toLocaleString() : 'N/A';

const TradeHistory = () => {
  const [history, setHistory] = useState([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selectedTrade, setSelectedTrade] = useState(null);

  const limit = 15;

  const fetchHistory = useCallback(async (pageNum) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/trade_history`, {
        params: { page: pageNum, limit },
      });
      const { trades, total } = response.data;
      setHistory(trades);
      setTotalPages(Math.ceil(total / limit));
    } catch (error) {
      console.error("Failed to fetch trade history:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory(page);
  }, [page, fetchHistory]);

  const handleNextPage = () => {
    if (page < totalPages) {
      setPage(page + 1);
    }
  };

  const handlePrevPage = () => {
    if (page > 1) {
      setPage(page - 1);
    }
  };

  const handleViewDetails = (trade) => {
    setSelectedTrade(trade);
  };

  return (
    <div className="card">
      <h2>Trade History</h2>
      <div style={{ overflowX: 'auto' }}>
        <table className="positions-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Symbol</th>
              <th>Type</th>
              <th>Amount (USD)</th>
              <th>Price</th>
              <th>P&L (USD)</th>
              <th>Reason</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan="7" style={{ textAlign: 'center' }}>Loading...</td></tr>
            ) : history.length > 0 ? (
              // Using index as a key is a fallback. A unique ID from the backend is preferable.
              history.map((trade) => {
                const pnlColor = trade.pnl >= 0 ? 'text-green' : 'text-red';
                return (
                  <tr key={trade.id}>
                    <td>{formatDate(trade.timestamp)}</td>
                    <td><strong>{trade.symbol}</strong></td>
                    <td>{trade.type.toUpperCase()}</td>
                    <td>{formatCurrency(trade.usdt_amount)}</td>
                    <td>{formatCurrency(trade.price)}</td>
                    <td className={pnlColor}>{formatCurrency(trade.pnl)}</td>
                    <td>{trade.reason}</td>
                    <td><button onClick={() => handleViewDetails(trade)}>Details</button></td>
                  </tr>
                );
              })
            ) : (
              <tr><td colSpan="7" style={{ textAlign: 'center' }}>No trade history found.</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="pagination-controls">
        <button onClick={handlePrevPage} disabled={page <= 1 || loading}>
          Previous
        </button>
        <span>Page {page} of {totalPages}</span>
        <button onClick={handleNextPage} disabled={page >= totalPages || loading}>
          Next
        </button>
      </div>
      {selectedTrade && <TradeDetailsModal trade={selectedTrade} onClose={() => setSelectedTrade(null)} />}
    </div>
  );
};

export default TradeHistory;