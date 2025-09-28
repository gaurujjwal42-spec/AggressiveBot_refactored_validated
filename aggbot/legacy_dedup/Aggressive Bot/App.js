import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import { io } from 'socket.io-client';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import './App.css';

import { NotificationProvider, useNotifications } from './context/NotificationContext';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import Configuration from './components/Configuration';
import TradeHistory from './components/TradeHistory';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

const API_URL = 'http://127.0.0.1:5000';

// This is the core component that handles data fetching and routing
const AppCore = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const { addNotification } = useNotifications();

  useEffect(() => {
    // Establish WebSocket connection
    const socket = io(API_URL);

    socket.on('connect', () => {
      console.log('WebSocket connected!');
      setError('');
    });

    socket.on('update_data', (liveData) => {
      setData(liveData);
    });

    socket.on('notification', (notification) => {
      addNotification(notification);
    });

    socket.on('disconnect', () => {
      console.error('WebSocket disconnected.');
      setError('Lost connection to the bot backend. Please refresh.');
    });

    // Cleanup on component unmount
    return () => {
      socket.disconnect();
    };
  }, [addNotification]);

  const handleControl = async (command) => {
    try {
      await axios.post(`${API_URL}/api/control`, { command });
    } catch (err) {
      setError(`Failed to ${command} the bot.`);
    }
  };

  const handleSell = async (positionId) => {
    if (!window.confirm(`Are you sure you want to sell position ${positionId}?`)) return;
    try {
      await axios.post(`${API_URL}/api/sell`, { position_id: positionId });
    } catch (err) {
      setError(`Failed to sell position ${positionId}.`);
    }
  };

  if (error) return <div className="App"><div className="status-bar status-ERROR">{error}</div></div>;
  if (!data) return <div className="App">Loading dashboard...</div>;
  
  const isPaused = data?.bot_status?.status === 'PAUSED';

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout handleControl={handleControl} isPaused={isPaused} />}>
          <Route index element={<Dashboard data={data} onSell={handleSell} />} />
          <Route path="config" element={<Configuration />} />
          <Route path="history" element={<TradeHistory />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

// --- Main App Component ---
// This component wraps the AppCore with the NotificationProvider
function App() {
  return (
    <NotificationProvider>
      <AppCore />
    </NotificationProvider>
  );
}

export default App;