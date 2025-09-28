#!/usr/bin/env python3
"""
Visualization tool for trading strategy optimization performance.

Reads performance data from the database and generates plots to analyze
the relationship between parameters and performance metrics.
"""

import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting

def load_performance_data(db_path="trading_bot.db") -> pd.DataFrame | None:
    """Load and process performance data from the database."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Load strategy_performance table
            df = pd.read_sql_query("SELECT * FROM strategy_performance", conn)

            if df.empty:
                print("No performance data found in the 'strategy_performance' table.")
                return None

            # Parse JSON columns
            params_df = pd.json_normalize(df['parameters'].apply(json.loads))
            metrics_df = pd.json_normalize(df['performance_metrics'].apply(json.loads))
            market_cond_df = pd.json_normalize(df['market_condition'].apply(json.loads))

            # Combine into a single DataFrame
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            performance_data = pd.concat(
                [df[['timestamp']], params_df, metrics_df, market_cond_df], axis=1
            )

            # Drop rows where essential metrics might be missing for plotting
            required_cols = [
                'profit_factor',
                'win_rate',
                'position_size_multiplier',
                'momentum_threshold',
                'regime'
            ]
            # Add a check for older data that might not have 'regime'
            if 'regime' not in performance_data.columns:
                print("Warning: 'regime' column not found. Regime comparison will be unavailable.")
                required_cols.remove('regime')
            performance_data.dropna(subset=required_cols, inplace=True)

            return performance_data

    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

def plot_parameter_vs_performance(data: pd.DataFrame, parameter: str, metric: str):
    """Create a scatter plot of a parameter vs. a performance metric."""
    if parameter not in data.columns or metric not in data.columns:
        print(f"Error: One or both of '{parameter}' or '{metric}' not found in data.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(data[parameter], data[metric], alpha=0.6, c=data['profit_factor'], cmap='viridis')
    plt.colorbar(label='Profit Factor')
    plt.title(f'Performance Analysis: {parameter.replace("_", " ").title()} vs {metric.replace("_", " ").title()}')
    plt.xlabel(parameter.replace("_", " ").title())
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_parameter_timeline(data: pd.DataFrame, parameter: str):
    """Plot how a parameter's value has changed over time."""
    if parameter not in data.columns:
        print(f"Error: Parameter '{parameter}' not found in data.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data[parameter], marker='o', linestyle='-', markersize=4)
    plt.title(f'Timeline of {parameter.replace("_", " ").title()} Optimization')
    plt.xlabel('Timestamp')
    plt.ylabel(parameter.replace("_", " ").title())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate() # Rotation
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_3d_parameter_performance(data: pd.DataFrame, param1: str, param2: str, metric: str):
    """Create a 3D scatter plot of two parameters vs. a performance metric."""
    if not all(p in data.columns for p in [param1, param2, metric]):
        print(f"Error: One or more of '{param1}', '{param2}', or '{metric}' not found in data.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D scatter plot
    scatter = ax.scatter(data[param1], data[param2], data[metric], c=data[metric], cmap='viridis', s=40)

    # Set labels and title
    ax.set_xlabel(param1.replace("_", " ").title())
    ax.set_ylabel(param2.replace("_", " ").title())
    ax.set_zlabel(metric.replace("_", " ").title())
    ax.set_title(f'3D Performance Analysis: {param1.title()} & {param2.title()} vs {metric.title()}')

    # Add a color bar
    fig.colorbar(scatter, shrink=0.5, aspect=5, label=metric.replace("_", " ").title())

    plt.tight_layout()
    plt.show()

def plot_regime_performance(data: pd.DataFrame, metric: str = 'profit_factor'):
    """Create a box plot comparing performance metrics across market regimes."""
    if metric not in data.columns or 'regime' not in data.columns:
        print(f"Error: Metric '{metric}' or 'regime' not found for regime comparison.")
        return

    # Filter for only 'TRENDING' and 'RANGING' regimes to avoid clutter
    regime_data = data[data['regime'].isin(['TRENDING', 'RANGING'])]

    if regime_data.empty or regime_data.groupby('regime').ngroups < 1:
        print("No data found for 'TRENDING' or 'RANGING' regimes to compare.")
        return

    plt.figure(figsize=(10, 6))

    # Prepare data for boxplot
    regime_values = [group[metric].dropna().tolist() for name, group in regime_data.groupby('regime')]
    regime_labels = [name for name, group in regime_data.groupby('regime')]

    plt.boxplot(regime_values, labels=regime_labels, patch_artist=True)
    plt.title(f'Performance Comparison by Market Regime: {metric.replace("_", " ").title()}')
    plt.xlabel('Market Regime')
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    print("Loading performance data for visualization...")
    performance_data = load_performance_data()

    if performance_data is None or performance_data.empty:
        print("Could not generate plots due to missing data.")
        return

    print("Data loaded successfully. Generating plots...")
    plot_parameter_vs_performance(performance_data, 'position_size_multiplier', 'profit_factor')
    plot_parameter_timeline(performance_data, 'take_profit_multiplier')

    # Add the new 3D plot
    plot_3d_parameter_performance(
        performance_data,
        'momentum_threshold',
        'position_size_multiplier',
        'profit_factor'
    )

    # Add the new regime comparison plot
    plot_regime_performance(performance_data, 'win_rate')

if __name__ == "__main__":
    main()