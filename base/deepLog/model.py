"""
DeepLog LSTM Model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


class DeepLogLSTM(nn.Module):
    """
    LSTM model for log sequence prediction.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(DeepLogLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.fc(last_output)  # (batch_size, input_size)
        return output


class DeepLog:
    """
    DeepLog anomaly detection model.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_candidates=9):
        """
        Initialize DeepLog model.
        
        Args:
            input_size: Size of vocabulary (number of unique events)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_candidates: Top-k candidates for anomaly detection
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_candidates = num_candidates
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepLogLSTM(input_size, hidden_size, num_layers).to(self.device)
        
        self.criterion = None
        self.optimizer = None
        
    def fit(self, X_train, y_train, epochs=50, batch_size=64, learning_rate=0.001, 
            validation_split=0.2, verbose=True):
        """
        Train the DeepLog model.
        
        Args:
            X_train: Training input windows
            y_train: Training target events
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
            verbose: Print training progress
            
        Returns:
            history: Dictionary with training history
        """
        # Split into train and validation
        n_val = int(len(X_train) * validation_split)
        indices = np.random.permutation(len(X_train))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(
            torch.LongTensor(X_train_split),
            torch.LongTensor(y_train_split)
        )
        val_dataset = TensorDataset(
            torch.LongTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        if verbose:
            print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<12}")
            print("-" * 50)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * correct / total
            
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # Print progress
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"{epoch+1:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_acc:<12.2f}")
        
        return history
    
    def predict(self, X_test):
        """
        Predict anomalies for test sequences.
        
        Args:
            X_test: Test input windows
            
        Returns:
            predictions: Boolean array (True = anomaly, False = normal)
        """
        self.model.eval()
        predictions = []
        
        # Convert to tensor
        X_test_tensor = torch.LongTensor(X_test).to(self.device)
        
        # Predict in batches
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = X_test_tensor[i:i + batch_size]
                outputs = self.model(batch)
                
                # Get top-k predictions
                probabilities = torch.softmax(outputs, dim=1)
                top_k_probs, top_k_indices = torch.topk(probabilities, self.num_candidates, dim=1)
                
                # Check if actual events would be in top-k
                # Since we don't have actual targets here, we return the top-k indices
                # The calling code will handle the comparison
                predictions.extend(top_k_indices.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_sequence(self, X_test, y_test):
        """
        Predict anomalies with actual targets for comparison.
        
        Args:
            X_test: Test input windows
            y_test: Actual target events
            
        Returns:
            predictions: Boolean array (True = anomaly, False = normal)
        """
        self.model.eval()
        predictions = []
        
        X_test_tensor = torch.LongTensor(X_test).to(self.device)
        
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_X = X_test_tensor[i:i + batch_size]
                batch_y = y_test[i:i + batch_size]
                
                outputs = self.model(batch_X)
                
                # Get top-k predictions
                probabilities = torch.softmax(outputs, dim=1)
                top_k_probs, top_k_indices = torch.topk(probabilities, self.num_candidates, dim=1)
                top_k_indices = top_k_indices.cpu().numpy()
                
                # Check if actual next event is in top-k
                for j, actual_event in enumerate(batch_y):
                    is_anomaly = actual_event not in top_k_indices[j]
                    predictions.append(is_anomaly)
        
        return np.array(predictions)
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_candidates': self.num_candidates
        }, filepath)
    
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.num_candidates = checkpoint['num_candidates']
        
        self.model = DeepLogLSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self
