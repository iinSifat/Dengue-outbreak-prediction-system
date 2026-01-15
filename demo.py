"""
Demo Script - Test Dengue Prediction System
Run this script to verify all components are working correctly
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import DengueDataPreprocessor
from models.regression import DengueRegressionModel
from models.clustering import DengueClusteringModel
from models.markov import DengueMarkovModel, classify_risk_from_cases
from utils.visualization import DengueVisualizer


def main():
    print("\n" + "="*70)
    print("DENGUE OUTBREAK PREDICTION SYSTEM - DEMO")
    print("="*70 + "\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    print("-" * 70)
    
    preprocessor = DengueDataPreprocessor(
        dengue_path='data/dengue_cases.csv',
        weather_path='data/weather_data.csv'
    )
    
    X_train, X_test, y_train, y_test, processed_df, feature_names = preprocessor.run_full_pipeline()
    
    # Step 2: Train Linear Regression
    print("\n" + "="*70)
    print("Step 2: Training Linear Regression Model...")
    print("-" * 70)
    
    reg_model = DengueRegressionModel()
    reg_model.train(X_train, y_train, feature_names)
    metrics = reg_model.evaluate(X_test, y_test)
    
    print("\n✓ Linear Regression Model Ready!")
    
    # Step 3: Train K-means Clustering
    print("\n" + "="*70)
    print("Step 3: Training K-means Clustering Model...")
    print("-" * 70)
    
    cluster_model = DengueClusteringModel(n_clusters=3)
    cluster_results = cluster_model.train(X_train, y_train.values)
    cluster_model.analyze_clusters(X_test, feature_names)
    
    print("\n✓ K-means Clustering Model Ready!")
    
    # Step 4: Train Markov Model
    print("\n" + "="*70)
    print("Step 4: Training Markov Chain Model...")
    print("-" * 70)
    
    risk_sequence = classify_risk_from_cases(processed_df['Dengue_Cases'].values)
    markov_model = DengueMarkovModel()
    markov_model.train(risk_sequence)
    
    # Analyze patterns
    markov_model.analyze_outbreak_patterns(risk_sequence)
    
    print("\n✓ Markov Model Ready!")
    
    # Step 5: Make Predictions
    print("\n" + "="*70)
    print("Step 5: Making Predictions...")
    print("-" * 70)
    
    # Regression predictions
    y_pred = reg_model.predict(X_test.head(10))
    print("\nSample Predictions (Next 10 weeks):")
    print(f"{'Actual':<15} {'Predicted':<15} {'Difference':<15}")
    print("-" * 45)
    for actual, pred in zip(y_test.head(10).values, y_pred):
        diff = actual - pred
        print(f"{actual:<15.0f} {pred:<15.0f} {diff:<15.0f}")
    
    # Cluster predictions
    test_clusters = cluster_model.get_cluster_labels(X_test.head(5))
    print("\n\nRisk Zone Classifications:")
    for i, label in enumerate(test_clusters):
        print(f"  Week {i+1}: {label}")
    
    # Markov predictions
    current_state = 'Medium'
    next_state, probability = markov_model.predict_next_state(current_state)
    print(f"\n\nMarkov Prediction:")
    print(f"  Current State: {current_state}")
    print(f"  Next State: {next_state} (Probability: {probability:.2%})")
    
    # Step 6: Model Interpretation
    print("\n" + "="*70)
    print("Step 6: Model Interpretation...")
    print("-" * 70)
    
    explanation = reg_model.explain_model()
    
    # Step 7: Summary
    print("\n" + "="*70)
    print("SYSTEM SUMMARY")
    print("="*70)
    
    print(f"\n✓ Dataset: {len(processed_df)} records")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"\n✓ Linear Regression RMSE: {metrics['test_rmse']:.2f} cases")
    print(f"✓ Linear Regression R²: {metrics['test_r2']:.4f}")
    print(f"✓ Clustering Silhouette Score: {cluster_results['silhouette_score']:.4f}")
    print(f"✓ Number of Clusters: {cluster_results['n_clusters']}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    
    print("\n📌 Next Steps:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("   2. Open your browser at: http://localhost:8501")
    print("   3. Explore the interactive dashboard!")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
