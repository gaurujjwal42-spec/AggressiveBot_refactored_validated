# -*- coding: utf-8 -*-
"""
Alert Dashboard Integration

Provides web interface for viewing and managing alerts from the advanced alert system.
Integrates with the existing monitoring dashboard to display real-time alert information.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from flask import Flask, render_template_string, jsonify, request
from advanced_alert_system import get_alert_system
from enhanced_logger import EnhancedLogger

class AlertDashboard:
    """Web dashboard for alert management"""
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.alert_system = get_alert_system()
        self.logger = EnhancedLogger()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize Flask app with alert routes"""
        self.app = app
        
        # Add alert routes
        app.add_url_rule('/alerts', 'alerts_dashboard', self.alerts_dashboard)
        app.add_url_rule('/api/alerts/summary', 'alert_summary', self.get_alert_summary)
        app.add_url_rule('/api/alerts/active', 'active_alerts', self.get_active_alerts)
        app.add_url_rule('/api/alerts/history', 'alert_history', self.get_alert_history)
        app.add_url_rule('/api/alerts/resolve/<alert_id>', 'resolve_alert', self.resolve_alert, methods=['POST'])
    
    def alerts_dashboard(self):
        """Render the alerts dashboard page"""
        template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot - Alert Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        .alerts-section {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .section-header {
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
            font-weight: bold;
        }
        .alert-item {
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .alert-item:last-child {
            border-bottom: none;
        }
        .alert-info {
            flex: 1;
        }
        .alert-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .alert-message {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .alert-time {
            color: #999;
            font-size: 0.8em;
        }
        .alert-severity {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .severity-critical {
            background: #dc3545;
            color: white;
        }
        .severity-high {
            background: #fd7e14;
            color: white;
        }
        .severity-medium {
            background: #ffc107;
            color: #212529;
        }
        .severity-low {
            background: #28a745;
            color: white;
        }
        .severity-info {
            background: #17a2b8;
            color: white;
        }
        .resolve-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 10px;
        }
        .resolve-btn:hover {
            background: #218838;
        }
        .no-alerts {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .refresh-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 Alert Dashboard</h1>
            <p>Real-time monitoring of trading bot alerts and system health</p>
        </div>
        
        <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
        
        <div class="stats-grid" id="statsGrid">
            <!-- Stats will be loaded here -->
        </div>
        
        <div class="alerts-section">
            <div class="section-header">
                Active Alerts
            </div>
            <div id="activeAlerts">
                <!-- Active alerts will be loaded here -->
            </div>
        </div>
        
        <div class="alerts-section" style="margin-top: 20px;">
            <div class="section-header">
                Recent Alert History
            </div>
            <div id="alertHistory">
                <!-- Alert history will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        function refreshData() {
            loadAlertSummary();
            loadActiveAlerts();
            loadAlertHistory();
        }
        
        function loadAlertSummary() {
            fetch('/api/alerts/summary')
                .then(response => response.json())
                .then(data => {
                    const statsGrid = document.getElementById('statsGrid');
                    statsGrid.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value">${data.active_alerts_count}</div>
                            <div class="stat-label">Active Alerts</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.active_by_severity.CRITICAL || 0}</div>
                            <div class="stat-label">Critical</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.active_by_severity.HIGH || 0}</div>
                            <div class="stat-label">High Priority</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.recent_alerts_count}</div>
                            <div class="stat-label">Last Hour</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.total_alerts_today}</div>
                            <div class="stat-label">Today Total</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.escalated_alerts}</div>
                            <div class="stat-label">Escalated</div>
                        </div>
                    `;
                })
                .catch(error => console.error('Error loading alert summary:', error));
        }
        
        function loadActiveAlerts() {
            fetch('/api/alerts/active')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('activeAlerts');
                    if (data.length === 0) {
                        container.innerHTML = '<div class="no-alerts">✅ No active alerts</div>';
                        return;
                    }
                    
                    container.innerHTML = data.map(alert => `
                        <div class="alert-item">
                            <div class="alert-info">
                                <div class="alert-title">${alert.title}</div>
                                <div class="alert-message">${alert.message}</div>
                                <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                            </div>
                            <div>
                                <span class="alert-severity severity-${alert.severity.toLowerCase()}">${alert.severity}</span>
                                <button class="resolve-btn" onclick="resolveAlert('${alert.id}')">Resolve</button>
                            </div>
                        </div>
                    `).join('');
                })
                .catch(error => console.error('Error loading active alerts:', error));
        }
        
        function loadAlertHistory() {
            fetch('/api/alerts/history?limit=10')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('alertHistory');
                    if (data.length === 0) {
                        container.innerHTML = '<div class="no-alerts">No recent alert history</div>';
                        return;
                    }
                    
                    container.innerHTML = data.map(alert => `
                        <div class="alert-item">
                            <div class="alert-info">
                                <div class="alert-title">${alert.title}</div>
                                <div class="alert-message">${alert.message}</div>
                                <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                            </div>
                            <div>
                                <span class="alert-severity severity-${alert.severity.toLowerCase()}">${alert.severity}</span>
                                ${alert.resolved ? '<span style="color: green; margin-left: 10px;">✓ Resolved</span>' : ''}
                            </div>
                        </div>
                    `).join('');
                })
                .catch(error => console.error('Error loading alert history:', error));
        }
        
        function resolveAlert(alertId) {
            fetch(`/api/alerts/resolve/${alertId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({resolution_note: 'Resolved via dashboard'})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    refreshData();
                } else {
                    alert('Failed to resolve alert: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error resolving alert:', error);
                alert('Failed to resolve alert');
            });
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>
        """
        return template
    
    def get_alert_summary(self):
        """Get alert summary statistics"""
        try:
            summary = self.alert_system.get_alert_summary()
            return jsonify(summary)
        except Exception as e:
            self.logger.log_error(f"Failed to get alert summary: {str(e)}", "DASHBOARD_ERROR")
            return jsonify({'error': str(e)}), 500
    
    def get_active_alerts(self):
        """Get list of active alerts"""
        try:
            active_alerts = []
            for alert in self.alert_system.active_alerts.values():
                active_alerts.append({
                    'id': alert.id,
                    'type': alert.type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'escalated': alert.escalated,
                    'context': alert.context
                })
            
            # Sort by severity and timestamp
            severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
            # Sort by timestamp first (newest to oldest), then by severity (most to least critical)
            active_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            active_alerts.sort(key=lambda x: severity_order.get(x['severity'], 5))

            return jsonify(active_alerts)
        except Exception as e:
            self.logger.log_error(f"Failed to get active alerts: {str(e)}", "DASHBOARD_ERROR")
            return jsonify({'error': str(e)}), 500
    
    def get_alert_history(self):
        """Get alert history"""
        try:
            limit = request.args.get('limit', 20, type=int)
            
            history = []
            for alert in list(self.alert_system.alert_history)[-limit:]:
                history.append({
                    'id': alert.id,
                    'type': alert.type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved,
                    'resolution_time': alert.resolution_time.isoformat() if alert.resolution_time else None,
                    'escalated': alert.escalated
                })
            
            # Sort by timestamp (newest first)
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return jsonify(history)
        except Exception as e:
            self.logger.log_error(f"Failed to get alert history: {str(e)}", "DASHBOARD_ERROR")
            return jsonify({'error': str(e)}), 500
    
    def resolve_alert(self, alert_id):
        """Resolve an alert"""
        try:
            data = request.get_json() or {}
            resolution_note = data.get('resolution_note', 'Resolved via dashboard')
            
            success = self.alert_system.resolve_alert(alert_id, resolution_note)
            
            if success:
                return jsonify({'success': True, 'message': 'Alert resolved successfully'})
            else:
                return jsonify({'success': False, 'error': 'Alert not found or already resolved'}), 404
                
        except Exception as e:
            self.logger.log_error(f"Failed to resolve alert {alert_id}: {str(e)}", "DASHBOARD_ERROR")
            return jsonify({'success': False, 'error': str(e)}), 500

# Global instance
_alert_dashboard = None

def get_alert_dashboard() -> AlertDashboard:
    """Get or create the global alert dashboard instance"""
    global _alert_dashboard
    if _alert_dashboard is None:
        _alert_dashboard = AlertDashboard()
    return _alert_dashboard

def init_alert_dashboard(app: Flask):
    """Initialize alert dashboard with Flask app"""
    dashboard = get_alert_dashboard()
    dashboard.init_app(app)
    return dashboard