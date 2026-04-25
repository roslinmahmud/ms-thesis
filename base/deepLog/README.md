# DeepLog Library

A custom implementation of DeepLog for log anomaly detection using LSTM networks.

## Components

### Preprocessor
Handles event encoding and sliding window creation for log sequences.

```python
from deeplog.preprocessor import Preprocessor

preprocessor = Preprocessor()
preprocessor.fit(sequences)  # Build vocabulary
X, y = preprocessor.transform(sequences, window_size=10)
```

### DeepLog Model
LSTM-based model for learning sequential patterns and detecting anomalies.

```python
from deeplog import DeepLog

model = DeepLog(
    input_size=vocab_size,
    hidden_size=128,
    num_layers=2,
    num_candidates=9
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    validation_split=0.2,
    verbose=True
)

# Predict
predictions = model.predict_sequence(X_test, y_test)
```

## Features

- **LSTM Architecture**: Multi-layer LSTM with embedding layer
- **Top-k Detection**: Configurable number of candidate predictions
- **Training History**: Returns loss and accuracy metrics
- **GPU Support**: Automatic CUDA detection and usage
- **Model Persistence**: Save and load trained models

## Requirements

- PyTorch
- NumPy

## Usage in baseline.ipynb

The notebook uses this library for:
1. Processing OAuth2 log sequences
2. Training on normal execution patterns
3. Detecting anomalous sequences based on top-k predictions
