"""
Dengue Outbreak Prediction System - Interactive Streamlit Application
A comprehensive AI-based system for predicting dengue outbreaks in Bangladesh
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import DengueDataPreprocessor
from models.regression import DengueRegressionModel, create_prediction_summary
from models.clustering import DengueClusteringModel
from models.markov import DengueMarkovModel, classify_risk_from_cases
from utils.visualization import DengueVisualizer

# Page configuration
st.set_page_config(
    page_title="Dengue Outbreak Prediction System",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .medium-risk {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_data(dengue_path, weather_path):
    """
    Load and preprocess data with caching
    """
    preprocessor = DengueDataPreprocessor(dengue_path, weather_path)
    X_train, X_test, y_train, y_test, processed_df, feature_names = preprocessor.run_full_pipeline()
    return X_train, X_test, y_train, y_test, processed_df, feature_names, preprocessor


@st.cache_resource
def train_models(X_train, y_train, feature_names, processed_df):
    """
    Train all models with caching
    """
    # Train Linear Regression
    reg_model = DengueRegressionModel()
    reg_model.train(X_train, y_train, feature_names)
    
    # Train K-means Clustering
    cluster_model = DengueClusteringModel(n_clusters=3)
    cluster_model.train(X_train, y_train.values)
    
    # Train Markov Model
    risk_sequence = classify_risk_from_cases(processed_df['Dengue_Cases'].values)
    markov_model = DengueMarkovModel()
    markov_model.train(risk_sequence)
    
    return reg_model, cluster_model, markov_model


def main():
    """
    Main application function
    """
    
    # Header
    st.markdown('<p class="main-header">🦟 Dengue Outbreak Prediction System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Early Warning System for Bangladesh</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # File upload option
    use_default_data = st.sidebar.checkbox("Use Default Dataset", value=True)
    
    if use_default_data:
        # Use absolute path relative to the app.py script
        app_dir = os.path.dirname(os.path.abspath(__file__))
        dengue_path = os.path.join(app_dir, "data/dengue_cases.csv")
        weather_path = os.path.join(app_dir, "data/weather_data.csv")
    else:
        st.sidebar.subheader("Upload Custom Data")
        dengue_file = st.sidebar.file_uploader("Upload Dengue Cases CSV", type=['csv'])
        weather_file = st.sidebar.file_uploader("Upload Weather Data CSV", type=['csv'])
        
        if dengue_file and weather_file:
            dengue_path = dengue_file
            weather_path = weather_file
        else:
            st.warning("⚠️ Please upload both datasets or use default data")
            return
    
    # Region selection
    region = st.sidebar.selectbox(
        "Select Region",
        ["Dhaka", "Chittagong", "Sylhet", "Khulna", "Rajshahi"]
    )
    
    # Prediction horizon (months)
    prediction_months = st.sidebar.slider(
        "Prediction Horizon (months)",
        min_value=1,
        max_value=12,
        value=3,
        step=1
    )
    
    st.sidebar.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔮 Predictions", 
        "🗺️ Risk Zones", 
        "📈 Analysis",
        "ℹ️ About"
    ])
    
    try:
        # Load data with region filtering
        with st.spinner("Loading and preprocessing data..."):
            # First load full data
            preprocessor = DengueDataPreprocessor(dengue_path, weather_path)
            X_train_full, X_test_full, y_train_full, y_test_full, processed_df_full, feature_names = preprocessor.run_full_pipeline()
            
            # Filter by selected region
            if region:
                region_mask = processed_df_full['Region'] == region
                processed_df = processed_df_full[region_mask].reset_index(drop=True)
                
                # Rebuild train/test split for this region
                n_train = int(len(processed_df) * 0.8)
                if n_train > 0 and len(processed_df) - n_train > 0:
                    X_train = X_train_full.iloc[:n_train]
                    X_test = X_test_full.iloc[n_train:n_train + len(processed_df) - n_train]
                    y_train = y_train_full.iloc[:n_train]
                    y_test = y_test_full.iloc[n_train:n_train + len(processed_df) - n_train]
                else:
                    X_train, X_test, y_train, y_test = X_train_full, X_test_full, y_train_full, y_test_full
            else:
                X_train, X_test, y_train, y_test = X_train_full, X_test_full, y_train_full, y_test_full
                processed_df = processed_df_full
        
        # Train models
        with st.spinner("Training AI models..."):
            reg_model, cluster_model, markov_model = train_models(
                X_train, y_train, feature_names, processed_df
            )
        
        # Visualizer
        viz = DengueVisualizer()
        
        # ========== TAB 1: OVERVIEW ==========
        with tab1:
            st.header("📊 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Records",
                    f"{len(processed_df):,}",
                    help="Total number of data points"
                )
            
            with col2:
                avg_cases = processed_df['Dengue_Cases'].mean()
                st.metric(
                    "Average Cases",
                    f"{avg_cases:.0f}",
                    help="Mean dengue cases per week"
                )
            
            with col3:
                max_cases = processed_df['Dengue_Cases'].max()
                st.metric(
                    "Peak Cases",
                    f"{max_cases:.0f}",
                    help="Highest recorded cases"
                )
            
            with col4:
                current_risk = processed_df['Risk_Level'].iloc[-1]
                st.metric(
                    "Current Risk",
                    current_risk,
                    help="Latest risk assessment"
                )
            
            st.markdown("---")
            
            # Dengue trend plot
            st.subheader("Historical Dengue Cases Trend")
            fig = viz.plot_dengue_trend(
                processed_df,
                title=f"Dengue Cases in {region} (2020-2024)"
            )
            st.pyplot(fig)
            plt.close()
            
            # Monthly distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Monthly Case Distribution")
                fig = viz.plot_monthly_distribution(processed_df)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Risk Level Timeline")
                fig = viz.plot_risk_timeline(processed_df)
                st.pyplot(fig)
                plt.close()
            
            # Dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(processed_df.tail(10), use_container_width=True)
        
        # ========== TAB 2: PREDICTIONS ==========
        with tab2:
            st.header("🔮 Dengue Outbreak Predictions")
            
            # Model evaluation
            st.subheader("Model Performance")
            
            # Evaluate models
            y_train_pred = reg_model.predict(X_train)
            y_test_pred = reg_model.predict(X_test)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
                st.metric("Training RMSE", f"{train_rmse:.2f} cases")
            
            with col2:
                test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
                st.metric("Test RMSE", f"{test_rmse:.2f} cases")
            
            with col3:
                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_test_pred)
                st.metric("R² Score", f"{r2:.4f}")
            
            st.markdown("---")
            
            # Actual vs Predicted
            st.subheader("Actual vs Predicted Cases")
            
            # Get test dates
            test_dates = processed_df['Date'].iloc[-len(y_test):].values
            
            fig = viz.plot_actual_vs_predicted(
                y_test.values, 
                y_test_pred, 
                dates=test_dates
            )
            st.pyplot(fig)
            plt.close()
            
            # Future predictions
            st.markdown("---")
            st.subheader(f"Future Predictions ({prediction_months} months ahead)")
            
            # Generate monthly predictions
            last_date = pd.to_datetime(processed_df['Date'].iloc[-1])
            future_dates = []
            for i in range(prediction_months):
                next_month = last_date + pd.DateOffset(months=i+1)
                future_dates.append(next_month)
            
            # Use last row features to predict forward
            if len(X_test) > 0:
                last_features = X_test.iloc[-1:].copy()
                # Repeat for weeks in prediction period
                n_weeks = prediction_months * 4
                future_features_list = [last_features.iloc[0:1] for _ in range(min(n_weeks, 52))]
                future_features = pd.concat(future_features_list, ignore_index=True) if future_features_list else X_test.iloc[-1:]
                weekly_predictions = reg_model.predict(future_features[:n_weeks] if len(future_features) >= n_weeks else future_features)
            else:
                weekly_predictions = reg_model.predict(X_test)
            
            # Convert weekly predictions to monthly averages
            monthly_predictions = []
            for m in range(prediction_months):
                start_idx = m * 4
                end_idx = min((m + 1) * 4, len(weekly_predictions))
                if start_idx < len(weekly_predictions):
                    month_avg = weekly_predictions[start_idx:end_idx].mean()
                    monthly_predictions.append(month_avg)
                else:
                    monthly_predictions.append(weekly_predictions[-1] if len(weekly_predictions) > 0 else 100)
            
            # Create prediction dataframe
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Cases': np.array(monthly_predictions).round(0).astype(int)
            })
            
            # Classify future risk
            future_risk = classify_risk_from_cases(np.array(monthly_predictions))
            future_df['Risk_Level'] = future_risk
            
            # Display predictions
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plot predictions
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical (last 20 weeks)
                hist_data = processed_df.tail(20)
                ax.plot(hist_data['Date'], hist_data['Dengue_Cases'], 
                       'b-o', linewidth=2, label='Historical', markersize=5)
                
                # Plot predictions
                ax.plot(future_df['Date'], future_df['Predicted_Cases'], 
                       'r--s', linewidth=2, label='Predicted', markersize=5)
                
                ax.set_xlabel('Date', fontweight='bold')
                ax.set_ylabel('Dengue Cases', fontweight='bold')
                ax.set_title('Dengue Case Forecast', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.dataframe(future_df, use_container_width=True)
                
                # Risk alert
                avg_future_risk = future_df['Predicted_Cases'].mean()
                if avg_future_risk > 400:
                    st.markdown(
                        '<div class="risk-box high-risk">⚠️ HIGH RISK ALERT<br>Expected: {:.0f} avg cases</div>'.format(avg_future_risk),
                        unsafe_allow_html=True
                    )
                elif avg_future_risk > 100:
                    st.markdown(
                        '<div class="risk-box medium-risk">⚡ MEDIUM RISK<br>Expected: {:.0f} avg cases</div>'.format(avg_future_risk),
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="risk-box low-risk">✅ LOW RISK<br>Expected: {:.0f} avg cases</div>'.format(avg_future_risk),
                        unsafe_allow_html=True
                    )
            
            # Feature importance
            st.markdown("---")
            st.subheader("Model Interpretation: Feature Importance")
            
            importance_df = reg_model.get_feature_importance(top_n=10)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = viz.plot_feature_importance(importance_df)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.write("**Top Contributing Factors:**")
                for idx, row in importance_df.head(5).iterrows():
                    impact = "↑ Increases" if row['Coefficient'] > 0 else "↓ Decreases"
                    st.write(f"• **{row['Feature']}**: {impact} cases")
        
        # ========== TAB 3: RISK ZONES ==========
        with tab3:
            st.header("🗺️ Risk Zone Classification")
            
            st.subheader("K-means Clustering Analysis")
            
            # Get cluster assignments
            cluster_labels = cluster_model.predict(X_test)
            risk_labels = cluster_model.get_cluster_labels(X_test)
            
            # Display cluster summary
            col1, col2, col3 = st.columns(3)
            
            risk_counts = pd.Series(risk_labels).value_counts()
            
            with col1:
                low_count = risk_counts.get('Low Risk', 0)
                low_pct = (low_count / len(risk_labels)) * 100
                st.metric("Low Risk Periods", f"{low_count}", f"{low_pct:.1f}%")
            
            with col2:
                med_count = risk_counts.get('Medium Risk', 0)
                med_pct = (med_count / len(risk_labels)) * 100
                st.metric("Medium Risk Periods", f"{med_count}", f"{med_pct:.1f}%")
            
            with col3:
                high_count = risk_counts.get('High Risk', 0)
                high_pct = (high_count / len(risk_labels)) * 100
                st.metric("High Risk Periods", f"{high_count}", f"{high_pct:.1f}%")
            
            st.markdown("---")
            
            # Cluster visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cluster Visualization")
                # Use temperature and rainfall for visualization
                if 'Temperature_C' in feature_names and 'Rainfall_mm' in feature_names:
                    temp_idx = feature_names.index('Temperature_C')
                    rain_idx = feature_names.index('Rainfall_mm')
                else:
                    temp_idx = 0
                    rain_idx = 1
                
                fig = viz.plot_cluster_visualization(
                    X_test.values,
                    cluster_labels,
                    feature1_idx=temp_idx,
                    feature2_idx=rain_idx,
                    feature_names=feature_names
                )
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Cluster Characteristics")
                centers_df = cluster_model.analyze_clusters(X_test, feature_names)
                st.dataframe(centers_df, use_container_width=True)
            
            # Markov state transitions
            st.markdown("---")
            st.subheader("Risk State Transitions (Markov Model)")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Transition Probability Matrix**")
                transition_df = pd.DataFrame(
                    markov_model.transition_matrix,
                    index=['From Low', 'From Medium', 'From High'],
                    columns=['To Low', 'To Medium', 'To High']
                )
                st.dataframe(transition_df.style.format("{:.2%}"), use_container_width=True)
            
            with col2:
                st.write("**Key Insights**")
                insights = markov_model.get_transition_insights()
                for insight in insights[:5]:
                    st.info(insight)
            
            # Current state prediction
            st.markdown("---")
            st.subheader("Next Week Risk Prediction")
            
            current_risk = processed_df['Risk_Level'].iloc[-1]
            next_state, probability = markov_model.predict_next_state(current_risk)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Current State:** {current_risk}")
            
            with col2:
                st.success(f"**Predicted Next State:** {next_state}")
            
            with col3:
                st.warning(f"**Confidence:** {probability:.1%}")
        
        # ========== TAB 4: ANALYSIS ==========
        with tab4:
            st.header("📈 Advanced Analysis")
            
            # Seasonal pattern
            st.subheader("Seasonal Dengue Pattern")
            fig = viz.plot_seasonal_pattern(processed_df)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            
            # Prediction error analysis
            st.subheader("Model Error Analysis")
            fig = viz.plot_prediction_error(y_test.values, y_test_pred)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            
            # Statistical summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Statistical Summary")
                summary_stats = processed_df[['Dengue_Cases', 'Temperature_C', 
                                             'Rainfall_mm', 'Humidity_Percent']].describe()
                st.dataframe(summary_stats, use_container_width=True)
            
            with col2:
                st.subheader("Risk Distribution")
                risk_dist = processed_df['Risk_Level'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#2ecc71', '#f39c12', '#e74c3c']
                ax.pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
                ax.set_title('Historical Risk Distribution', fontweight='bold')
                st.pyplot(fig)
                plt.close()
        
        # ========== TAB 5: ABOUT ==========
        with tab5:
            st.header("ℹ️ About This System")
            
            st.markdown("""
            ### Dengue Outbreak Prediction System for Bangladesh
            
            This AI-powered system predicts dengue outbreak trends and provides early warning 
            insights to support public health planning in Bangladesh.
            
            #### 🎯 Objectives
            - Predict future dengue case trends (weekly/monthly)
            - Identify high-risk periods and regions
            - Support early outbreak warnings
            - Provide explainable AI insights
            
            #### 🤖 Machine Learning Algorithms
            
            **1. Linear Regression**
            - Predicts future dengue case counts based on environmental factors
            - Provides feature importance for interpretation
            - RMSE and R² metrics for model evaluation
            
            **2. K-means Clustering**
            - Clusters time periods into risk zones (Low/Medium/High)
            - Identifies patterns in outbreak characteristics
            - Helps in resource allocation planning
            
            **3. Markov Chain Model**
            - Models transitions between risk states
            - Predicts next-week risk level
            - Provides probability-based forecasts
            
            #### 📊 Key Features
            - Historical data visualization
            - Multi-week ahead predictions
            - Risk zone classification
            - Seasonal pattern analysis
            - Model explainability
            
            #### 🗂️ Data Sources
            - Historical dengue case counts (2020-2024)
            - Weather data (temperature, rainfall, humidity)
            - Time features (month, season, monsoon indicators)
            
            #### ⚠️ Limitations
            - Predictions based on historical patterns
            - Weather forecast accuracy affects future predictions
            - Does not account for intervention measures
            - Designed for academic and planning purposes
            
            #### 👨‍💻 Technical Stack
            - **Language:** Python
            - **ML Libraries:** scikit-learn, numpy, pandas
            - **Visualization:** matplotlib, seaborn
            - **UI Framework:** Streamlit
            
            #### 📝 Usage Guidelines
            1. Select region and prediction horizon
            2. Review historical data and trends
            3. Analyze model predictions
            4. Interpret risk levels and take action
            5. Monitor weekly for updates
            
            ---
            
            **Developed for Academic Research & Public Health Planning**
            
            *Bangladesh Dengue Surveillance Initiative*
            """)
            
            st.success("✅ System Status: All models operational")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
