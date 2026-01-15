"""
Markov Model for Dengue Outbreak State Transitions
Models transitions between different outbreak risk states (Low → Medium → High)
"""

import numpy as np
import pandas as pd
import pickle


class DengueMarkovModel:
    """
    Markov Chain model for dengue outbreak state transitions
    States: Low Risk, Medium Risk, High Risk
    """
    
    def __init__(self, states=['Low', 'Medium', 'High']):
        """
        Initialize Markov model
        
        Args:
            states: List of outbreak states
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_index = {state: idx for idx, state in enumerate(states)}
        self.index_to_state = {idx: state for idx, state in enumerate(states)}
        
        # Transition matrix: P[i][j] = probability of moving from state i to state j
        self.transition_matrix = None
        self.is_trained = False
        
    def train(self, risk_sequence):
        """
        Train Markov model by computing transition probabilities
        
        Args:
            risk_sequence: Sequence of risk states (e.g., ['Low', 'Medium', 'High', ...])
            
        Returns:
            np.array: Transition probability matrix
        """
        print("\n" + "="*60)
        print("TRAINING MARKOV MODEL")
        print("="*60 + "\n")
        
        # Initialize transition count matrix
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for i in range(len(risk_sequence) - 1):
            current_state = risk_sequence[i]
            next_state = risk_sequence[i + 1]
            
            if current_state in self.state_to_index and next_state in self.state_to_index:
                current_idx = self.state_to_index[current_state]
                next_idx = self.state_to_index[next_state]
                transition_counts[current_idx][next_idx] += 1
        
        # Convert counts to probabilities
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] = transition_counts[i] / row_sum
            else:
                # If no transitions from this state, assume uniform distribution
                self.transition_matrix[i] = np.ones(self.n_states) / self.n_states
        
        self.is_trained = True
        
        print("✓ Transition matrix computed")
        self._display_transition_matrix()
        
        return self.transition_matrix
    
    def _display_transition_matrix(self):
        """
        Display the transition probability matrix in readable format
        """
        print("\nTransition Probability Matrix:")
        print("(Rows = Current State, Columns = Next State)\n")
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(
            self.transition_matrix,
            index=[f"From {state}" for state in self.states],
            columns=[f"To {state}" for state in self.states]
        )
        
        print(df.to_string())
        print()
        
        # Interpret transitions
        print("Interpretation:")
        for i, current_state in enumerate(self.states):
            for j, next_state in enumerate(self.states):
                prob = self.transition_matrix[i][j]
                if prob > 0:
                    print(f"  {current_state} → {next_state}: {prob:.2%}")
    
    def predict_next_state(self, current_state):
        """
        Predict the most likely next state
        
        Args:
            current_state: Current risk state
            
        Returns:
            tuple: (predicted_state, probability)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if current_state not in self.state_to_index:
            raise ValueError(f"Unknown state: {current_state}")
        
        current_idx = self.state_to_index[current_state]
        next_state_probs = self.transition_matrix[current_idx]
        
        # Get most likely next state
        next_idx = np.argmax(next_state_probs)
        next_state = self.index_to_state[next_idx]
        probability = next_state_probs[next_idx]
        
        return next_state, probability
    
    def predict_sequence(self, initial_state, steps=5):
        """
        Predict a sequence of future states
        
        Args:
            initial_state: Starting risk state
            steps: Number of steps to predict
            
        Returns:
            list: Sequence of predicted states
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        sequence = [initial_state]
        current_state = initial_state
        
        for _ in range(steps):
            next_state, _ = self.predict_next_state(current_state)
            sequence.append(next_state)
            current_state = next_state
        
        return sequence
    
    def get_state_probabilities(self, current_state):
        """
        Get probability distribution for next state
        
        Args:
            current_state: Current risk state
            
        Returns:
            dict: Probabilities for each possible next state
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if current_state not in self.state_to_index:
            raise ValueError(f"Unknown state: {current_state}")
        
        current_idx = self.state_to_index[current_state]
        next_state_probs = self.transition_matrix[current_idx]
        
        return {state: float(next_state_probs[idx]) 
                for state, idx in self.state_to_index.items()}
    
    def get_steady_state(self, iterations=100):
        """
        Calculate steady-state probabilities (long-term behavior)
        
        Args:
            iterations: Number of iterations for convergence
            
        Returns:
            dict: Steady-state probability for each state
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Start with uniform distribution
        state_vector = np.ones(self.n_states) / self.n_states
        
        # Iterate until convergence
        for _ in range(iterations):
            state_vector = state_vector @ self.transition_matrix
        
        return {state: float(state_vector[idx]) 
                for state, idx in self.state_to_index.items()}
    
    def analyze_outbreak_patterns(self, risk_sequence):
        """
        Analyze patterns in outbreak sequences
        
        Args:
            risk_sequence: Sequence of risk states
            
        Returns:
            dict: Pattern analysis
        """
        print("\n" + "="*60)
        print("OUTBREAK PATTERN ANALYSIS")
        print("="*60 + "\n")
        
        # Count state occurrences
        state_counts = {state: 0 for state in self.states}
        for state in risk_sequence:
            if state in state_counts:
                state_counts[state] += 1
        
        total = len(risk_sequence)
        
        print("State Distribution:")
        for state, count in state_counts.items():
            percentage = (count / total) * 100
            print(f"  {state}: {count} occurrences ({percentage:.1f}%)")
        
        # Find longest consecutive periods
        print("\nLongest Consecutive Periods:")
        for state in self.states:
            max_consecutive = 0
            current_consecutive = 0
            
            for s in risk_sequence:
                if s == state:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            print(f"  {state}: {max_consecutive} weeks")
        
        # Calculate steady state
        if self.is_trained:
            steady_state = self.get_steady_state()
            print("\nSteady-State Probabilities (Long-term Expected Distribution):")
            for state, prob in steady_state.items():
                print(f"  {state}: {prob:.2%}")
        
        return {
            'state_counts': state_counts,
            'total_periods': total,
            'steady_state': self.get_steady_state() if self.is_trained else None
        }
    
    def predict_with_confidence(self, current_state, confidence_threshold=0.5):
        """
        Predict next state with confidence assessment
        
        Args:
            current_state: Current risk state
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            dict: Prediction with confidence information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        next_state, probability = self.predict_next_state(current_state)
        
        # Get all probabilities
        all_probs = self.get_state_probabilities(current_state)
        
        # Determine confidence level
        if probability >= 0.7:
            confidence = "High"
        elif probability >= confidence_threshold:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            'current_state': current_state,
            'predicted_next_state': next_state,
            'probability': probability,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def simulate_outbreak_scenario(self, initial_state, weeks=12):
        """
        Simulate possible outbreak scenarios
        
        Args:
            initial_state: Starting state
            weeks: Number of weeks to simulate
            
        Returns:
            list: Simulated sequence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        sequence = [initial_state]
        current_state = initial_state
        
        for week in range(weeks):
            # Get transition probabilities
            current_idx = self.state_to_index[current_state]
            probs = self.transition_matrix[current_idx]
            
            # Sample next state based on probabilities
            next_idx = np.random.choice(self.n_states, p=probs)
            next_state = self.index_to_state[next_idx]
            
            sequence.append(next_state)
            current_state = next_state
        
        return sequence
    
    def get_transition_insights(self):
        """
        Get insights about state transitions
        
        Returns:
            dict: Transition insights
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        insights = []
        
        for i, current_state in enumerate(self.states):
            for j, next_state in enumerate(self.states):
                prob = self.transition_matrix[i][j]
                
                if prob > 0.5:
                    insights.append(f"• From {current_state}, very likely to stay/move to {next_state} ({prob:.1%})")
                elif prob > 0.3:
                    insights.append(f"• From {current_state}, moderate chance to move to {next_state} ({prob:.1%})")
        
        return insights
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model file
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'states': self.states,
            'transition_matrix': self.transition_matrix,
            'state_to_index': self.state_to_index,
            'index_to_state': self.index_to_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Markov model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.states = model_data['states']
        self.transition_matrix = model_data['transition_matrix']
        self.state_to_index = model_data['state_to_index']
        self.index_to_state = model_data['index_to_state']
        self.n_states = len(self.states)
        self.is_trained = True
        
        print(f"✓ Markov model loaded from {filepath}")


def classify_risk_from_cases(dengue_cases, low_threshold=100, high_threshold=400):
    """
    Convert dengue case counts to risk categories
    
    Args:
        dengue_cases: Array or Series of case counts
        low_threshold: Cases below this are Low risk
        high_threshold: Cases above this are High risk
        
    Returns:
        list: Risk categories
    """
    risk_levels = []
    
    for cases in dengue_cases:
        if cases < low_threshold:
            risk_levels.append('Low')
        elif cases < high_threshold:
            risk_levels.append('Medium')
        else:
            risk_levels.append('High')
    
    return risk_levels


if __name__ == "__main__":
    # Example usage
    print("Markov Model for Dengue Outbreak State Transitions")
    print("This module models transitions between risk states")
    print("\nExample:")
    print("  model = DengueMarkovModel()")
    print("  model.train(risk_sequence)")
    print("  next_state, prob = model.predict_next_state('Medium')")
    print("  future_seq = model.predict_sequence('Low', steps=5)")
