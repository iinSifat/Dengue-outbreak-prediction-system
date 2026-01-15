"""
Linear Regression Model for Dengue Case Prediction
Predicts future dengue case counts based on historical data and environmental factors
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle


class DengueRegressionModel:
    """
    Linear Regression model for dengue outbreak prediction
    """
    
    def __init__(self):
        """
        Initialize the regression model
        """
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_names = None
        self.metrics = {}
        
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the linear regression model
        
        Args:
            X_train: Training features
            y_train: Training target (dengue cases)
            feature_names: List of feature names for interpretation
            
        Returns:
            dict: Training metrics
        """
        print("\n" + "="*60)
        print("TRAINING LINEAR REGRESSION MODEL")
        print("="*60 + "\n")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names
        self.is_trained = True
        
        # Make predictions on training data
        y_train_pred = self.model.predict(X_train)
        
        # Calculate training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        self.metrics['train_rmse'] = train_rmse
        self.metrics['train_mae'] = train_mae
        self.metrics['train_r2'] = train_r2
        
        print(f"✓ Model trained successfully")
        print(f"\nTraining Performance:")
        print(f"  RMSE: {train_rmse:.2f} cases")
        print(f"  MAE:  {train_mae:.2f} cases")
        print(f"  R² Score: {train_r2:.4f}")
        
        return self.metrics
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test target (actual dengue cases)
            
        Returns:
            dict: Test metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST DATA")
        print("="*60 + "\n")
        
        # Make predictions
        y_test_pred = self.model.predict(X_test)
        
        # Calculate test metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.metrics['test_rmse'] = test_rmse
        self.metrics['test_mae'] = test_mae
        self.metrics['test_r2'] = test_r2
        
        print(f"Test Performance:")
        print(f"  RMSE: {test_rmse:.2f} cases")
        print(f"  MAE:  {test_mae:.2f} cases")
        print(f"  R² Score: {test_r2:.4f}")
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        print(f"  MAPE: {mape:.2f}%")
        
        return self.metrics
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Feature data for prediction
            
        Returns:
            array: Predicted dengue cases
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance based on regression coefficients
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame: Feature importance data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.model.coef_))]
        else:
            feature_names = self.feature_names
        
        # Get absolute coefficients for importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_,
            'Abs_Coefficient': np.abs(self.model.coef_)
        })
        
        # Sort by absolute coefficient
        importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
        
        return importance_df.head(top_n)
    
    def explain_model(self):
        """
        Provide explainable insights about the model
        
        Returns:
            dict: Explanation dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        print("\n" + "="*60)
        print("MODEL EXPLANATION & INTERPRETATION")
        print("="*60 + "\n")
        
        print(f"Model Type: Linear Regression")
        print(f"Intercept: {self.model.intercept_:.2f}")
        print(f"\nTop 10 Most Important Features:")
        
        importance_df = self.get_feature_importance(top_n=10)
        
        for idx, row in importance_df.iterrows():
            impact = "increases" if row['Coefficient'] > 0 else "decreases"
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f} ({impact} cases)")
        
        explanation = {
            'intercept': self.model.intercept_,
            'coefficients': dict(zip(self.feature_names, self.model.coef_)),
            'top_features': importance_df.to_dict('records'),
            'model_equation': self._generate_equation()
        }
        
        return explanation
    
    def _generate_equation(self):
        """
        Generate human-readable model equation
        
        Returns:
            str: Model equation
        """
        if self.feature_names is None or len(self.feature_names) == 0:
            return "Equation not available"
        
        equation = f"Dengue Cases = {self.model.intercept_:.2f}"
        
        # Show top 5 features in equation
        importance_df = self.get_feature_importance(top_n=5)
        
        for idx, row in importance_df.iterrows():
            sign = "+" if row['Coefficient'] > 0 else "-"
            equation += f" {sign} {abs(row['Coefficient']):.2f} × {row['Feature']}"
        
        equation += " + ... (other features)"
        
        return equation
    
    def predict_future(self, future_features, weeks_ahead=4):
        """
        Predict dengue cases for future time periods
        
        Args:
            future_features: Features for future predictions
            weeks_ahead: Number of weeks to predict
            
        Returns:
            array: Future predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        predictions = self.predict(future_features)
        
        print(f"\n✓ Generated predictions for {weeks_ahead} weeks ahead")
        
        return predictions
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model file
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.is_trained = True
        
        print(f"✓ Model loaded from {filepath}")


def create_prediction_summary(actual, predicted, dates=None):
    """
    Create a summary DataFrame comparing actual vs predicted values
    
    Args:
        actual: Actual dengue cases
        predicted: Predicted dengue cases
        dates: Optional date index
        
    Returns:
        DataFrame: Comparison summary
    """
    summary = pd.DataFrame({
        'Actual_Cases': actual,
        'Predicted_Cases': predicted,
        'Difference': actual - predicted,
        'Percentage_Error': ((actual - predicted) / actual * 100)
    })
    
    if dates is not None:
        summary['Date'] = dates
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Linear Regression Model for Dengue Prediction")
    print("This module should be used with preprocessed data")
    print("\nExample:")
    print("  model = DengueRegressionModel()")
    print("  model.train(X_train, y_train, feature_names)")
    print("  model.evaluate(X_test, y_test)")
    print("  predictions = model.predict(X_new)")
