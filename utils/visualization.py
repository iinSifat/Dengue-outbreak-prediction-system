"""
Visualization Module for Dengue Outbreak Prediction System
Provides charts and graphs for data analysis and model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime


# Set style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DengueVisualizer:
    """
    Main visualization class for dengue outbreak analysis
    """
    
    def __init__(self):
        """
        Initialize visualizer
        """
        self.color_map = {
            'Low': '#2ecc71',      # Green
            'Medium': '#f39c12',   # Orange
            'High': '#e74c3c'      # Red
        }
    
    def plot_dengue_trend(self, df, date_column='Date', cases_column='Dengue_Cases', 
                         title='Dengue Cases Over Time', save_path=None):
        """
        Plot dengue case trend over time
        
        Args:
            df: DataFrame with date and case data
            date_column: Name of date column
            cases_column: Name of dengue cases column
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(df[date_column], df[cases_column], 
               linewidth=2, color='#3498db', label='Dengue Cases')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_actual_vs_predicted(self, actual, predicted, dates=None, 
                                title='Actual vs Predicted Dengue Cases', save_path=None):
        """
        Plot actual vs predicted dengue cases
        
        Args:
            actual: Actual case counts
            predicted: Predicted case counts
            dates: Optional date index
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = dates if dates is not None else range(len(actual))
        
        ax.plot(x, actual, linewidth=2, color='#2c3e50', 
               label='Actual Cases', marker='o', markersize=3)
        ax.plot(x, predicted, linewidth=2, color='#e74c3c', 
               label='Predicted Cases', linestyle='--', marker='s', markersize=3)
        
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if dates is not None:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_distribution(self, df, month_column='Month', cases_column='Dengue_Cases',
                                 title='Monthly Dengue Case Distribution', save_path=None):
        """
        Plot monthly case distribution as bar chart
        
        Args:
            df: DataFrame with month and case data
            month_column: Name of month column
            cases_column: Name of cases column
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate average cases per month
        monthly_avg = df.groupby(month_column)[cases_column].mean()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        colors = ['#3498db' if val < monthly_avg.mean() else '#e74c3c' 
                 for val in monthly_avg.values]
        
        bars = ax.bar(range(1, 13), monthly_avg.values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Cases', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_risk_timeline(self, df, date_column='Date', risk_column='Risk_Level',
                          title='Dengue Outbreak Risk Level Timeline', save_path=None):
        """
        Plot risk level changes over time
        
        Args:
            df: DataFrame with date and risk level
            date_column: Name of date column
            risk_column: Name of risk level column
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Map risk levels to numeric values
        risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['Risk_Numeric'] = df[risk_column].map(risk_map)
        
        # Create color segments
        for risk_level, color in self.color_map.items():
            mask = df[risk_column] == risk_level
            if mask.any():
                ax.scatter(df[mask][date_column], df[mask]['Risk_Numeric'], 
                          color=color, label=risk_level, s=50, alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Risk Level', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Low', 'Medium', 'High'])
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cluster_visualization(self, X, cluster_labels, feature1_idx=0, feature2_idx=1,
                                  feature_names=None, title='K-means Clustering Results',
                                  save_path=None):
        """
        Visualize clustering results
        
        Args:
            X: Feature data
            cluster_labels: Cluster assignments
            feature1_idx: Index of first feature for visualization
            feature2_idx: Index of second feature for visualization
            feature_names: Names of features
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db']
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            ax.scatter(X[mask, feature1_idx], X[mask, feature2_idx],
                      c=colors[cluster_id % len(colors)], 
                      label=f'Cluster {cluster_id}',
                      s=100, alpha=0.6, edgecolors='black')
        
        if feature_names:
            ax.set_xlabel(feature_names[feature1_idx], fontsize=12, fontweight='bold')
            ax.set_ylabel(feature_names[feature2_idx], fontsize=12, fontweight='bold')
        else:
            ax.set_xlabel(f'Feature {feature1_idx}', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Feature {feature2_idx}', fontsize=12, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, importance_df, title='Feature Importance',
                               top_n=10, save_path=None):
        """
        Plot feature importance from regression model
        
        Args:
            importance_df: DataFrame with features and coefficients
            title: Plot title
            top_n: Number of top features to show
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top features
        plot_data = importance_df.head(top_n).copy()
        
        # Create horizontal bar chart
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in plot_data['Coefficient']]
        
        bars = ax.barh(range(len(plot_data)), plot_data['Coefficient'].values, 
                       color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data['Feature'].values)
        ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, df, features, title='Feature Correlation Heatmap',
                                save_path=None):
        """
        Plot correlation heatmap of features
        
        Args:
            df: DataFrame with features
            features: List of feature columns
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_seasonal_pattern(self, df, month_column='Month', cases_column='Dengue_Cases',
                             title='Seasonal Dengue Pattern', save_path=None):
        """
        Plot seasonal pattern with shaded regions
        
        Args:
            df: DataFrame with month and cases
            month_column: Name of month column
            cases_column: Name of cases column
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Calculate average and std per month
        monthly_stats = df.groupby(month_column)[cases_column].agg(['mean', 'std'])
        
        months = monthly_stats.index
        means = monthly_stats['mean'].values
        stds = monthly_stats['std'].values
        
        # Plot mean line
        ax.plot(months, means, linewidth=3, color='#3498db', marker='o', 
               markersize=8, label='Average Cases')
        
        # Add confidence band
        ax.fill_between(months, means - stds, means + stds, 
                       alpha=0.3, color='#3498db', label='±1 Std Dev')
        
        # Shade seasonal periods
        ax.axvspan(3, 5, alpha=0.2, color='#f39c12', label='Pre-Monsoon')
        ax.axvspan(6, 9, alpha=0.2, color='#e74c3c', label='Monsoon')
        ax.axvspan(10, 11, alpha=0.2, color='#f39c12', label='Post-Monsoon')
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dengue Cases', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_error(self, actual, predicted, title='Prediction Error Analysis',
                            save_path=None):
        """
        Plot prediction error distribution
        
        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        errors = actual - predicted
        
        # Error histogram
        axes[0].hist(errors, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        axes[1].scatter(predicted, errors, alpha=0.6, color='#e74c3c', s=50)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Cases', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Residual Error', fontsize=11, fontweight='bold')
        axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard_summary(self, df, predictions=None, save_path=None):
        """
        Create a comprehensive dashboard with multiple plots
        
        Args:
            df: Main dataframe
            predictions: Optional predictions data
            save_path: Optional path to save figure
            
        Returns:
            matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Dengue Trend
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['Date'], df['Dengue_Cases'], linewidth=2, color='#3498db')
        ax1.set_title('Dengue Cases Trend', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cases')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Monthly Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        monthly_avg = df.groupby('Month')['Dengue_Cases'].mean()
        ax2.bar(range(1, 13), monthly_avg.values, color='#e74c3c', alpha=0.7)
        ax2.set_title('Monthly Average Cases', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Average Cases')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Risk Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if 'Risk_Level' in df.columns:
            risk_counts = df['Risk_Level'].value_counts()
            colors_list = [self.color_map.get(level, '#95a5a6') for level in risk_counts.index]
            ax3.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=colors_list, startangle=90)
            ax3.set_title('Risk Level Distribution', fontsize=12, fontweight='bold')
        
        # Plot 4: Seasonal Pattern
        ax4 = fig.add_subplot(gs[2, :])
        monthly_stats = df.groupby('Month')['Dengue_Cases'].agg(['mean', 'std'])
        ax4.plot(monthly_stats.index, monthly_stats['mean'], linewidth=2, 
                marker='o', color='#2ecc71')
        ax4.fill_between(monthly_stats.index, 
                        monthly_stats['mean'] - monthly_stats['std'],
                        monthly_stats['mean'] + monthly_stats['std'],
                        alpha=0.3, color='#2ecc71')
        ax4.set_title('Seasonal Pattern with Variability', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Cases')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Dengue Outbreak Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    print("Dengue Outbreak Visualization Module")
    print("Provides comprehensive visualization tools for analysis")
    print("\nExample:")
    print("  viz = DengueVisualizer()")
    print("  viz.plot_dengue_trend(df)")
    print("  viz.plot_actual_vs_predicted(actual, predicted)")
