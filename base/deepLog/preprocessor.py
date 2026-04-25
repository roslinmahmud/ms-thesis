"""
Preprocessor for DeepLog
Handles event encoding and sliding window creation
"""

import numpy as np


class Preprocessor:
    """
    Preprocessor for log sequences.
    Handles event vocabulary creation, encoding, and sliding window generation.
    """
    
    def __init__(self):
        self.event_to_id = {}
        self.id_to_event = {}
        self.num_classes = 0
    
    def fit(self, sequences):
        """
        Build vocabulary from training sequences.
        
        Args:
            sequences: List of event sequences (each sequence is a list of events)
        """
        # Collect all unique events
        unique_events = set()
        for seq in sequences:
            unique_events.update(seq)
        
        # Sort for consistent ordering
        unique_events = sorted(list(unique_events))
        
        # Create mappings
        self.event_to_id = {event: idx for idx, event in enumerate(unique_events)}
        self.id_to_event = {idx: event for event, idx in self.event_to_id.items()}
        self.num_classes = len(self.event_to_id)
        
        return self
    
    def transform(self, sequences, window_size=10):
        """
        Transform sequences into sliding windows for training.
        
        Args:
            sequences: List of event sequences
            window_size: Size of sliding window
            
        Returns:
            X: Input windows (numpy array)
            y: Target events (numpy array)
        """
        X = []
        y = []
        
        for sequence in sequences:
            # Create sliding windows
            for i in range(len(sequence) - window_size):
                window = sequence[i:i + window_size]
                target = sequence[i + window_size]
                
                # Encode events
                try:
                    window_encoded = [self.event_to_id[event] for event in window]
                    target_encoded = self.event_to_id[target]
                    
                    X.append(window_encoded)
                    y.append(target_encoded)
                except KeyError:
                    # Skip if event not in vocabulary
                    continue
        
        return np.array(X), np.array(y)
    
    def fit_transform(self, sequences, window_size=10):
        """
        Fit and transform in one step.
        
        Args:
            sequences: List of event sequences
            window_size: Size of sliding window
            
        Returns:
            X: Input windows
            y: Target events
        """
        self.fit(sequences)
        return self.transform(sequences, window_size)
