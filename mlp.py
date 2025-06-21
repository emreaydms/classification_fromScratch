import numpy as np
from typing import List, Tuple, Optional, Union


class MLP:
    def __init__(
            self,
            input_size: int,
            hidden_sizes: List[int],
            output_size: int,
            activation: str = 'relu',
            output_activation: str = 'softmax',
            learning_rate: float = 0.01
    ):
        """
        Initialize Multi-Layer Perceptron

        Args:
            input_size: Number of input features
            hidden_sizes: List of sizes for hidden layers
            output_size: Number of output classes
            activation: Activation function for hidden layers ('relu' or 'sigmoid')
            output_activation: Activation function for output layer ('softmax' or 'sigmoid')
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation

        self.weights = []
        self.biases = []

        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * 0.01)
        self.biases.append(np.zeros((1, hidden_sizes[0])))

        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, hidden_sizes[i + 1])))

        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-x))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _activation_derivative(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Compute derivative of activation function"""
        if activation == 'sigmoid':
            sig = self._sigmoid(x)
            return sig * (1 - sig)
        elif activation == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Tuple of (activations, pre_activations) for all layers
        """
        activations = [X]
        pre_activations = []

        a = X
        for i in range(len(self.hidden_sizes)):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if self.activation == 'relu':
                a = self._relu(z)
            else:
                a = self._sigmoid(z)
            activations.append(a)

        z_out = a @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z_out)
        if self.output_activation == 'softmax':
            a_out = self._softmax(z_out)
        else:
            a_out = self._sigmoid(z_out)
        activations.append(a_out)

        return activations, pre_activations

    def backward(
            self,
            X: np.ndarray,
            y: np.ndarray,
            activations: List[np.ndarray],
            pre_activations: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass to compute gradients

        Args:
            X: Input data
            y: True labels (integer vector or one-hot matrix)
            activations: List of activations from forward pass
            pre_activations: List of pre-activations from forward pass

        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = X.shape[0]
        weight_grads = [None] * len(self.weights)
        bias_grads = [None] * len(self.biases)

        if self.output_activation == 'softmax':
            if y.ndim == 1:
                y_onehot = np.zeros_like(activations[-1])
                y_onehot[np.arange(m), y] = 1
            else:
                y_onehot = y
            delta = (activations[-1] - y_onehot) / m
        else:
            if y.ndim == 2:
                y_vec = y.reshape(-1, 1)
            else:
                y_vec = y.reshape(-1, 1)
            delta = (activations[-1] - y_vec) / m

        weight_grads[-1] = activations[-2].T @ delta
        bias_grads[-1] = np.sum(delta, axis=0, keepdims=True)

        for l in reversed(range(len(self.hidden_sizes))):
            delta = (delta @ self.weights[l + 1].T) * self._activation_derivative(pre_activations[l], self.activation)
            weight_grads[l] = activations[l].T @ delta
            bias_grads[l] = np.sum(delta, axis=0, keepdims=True)

        return weight_grads, bias_grads

    def update_parameters(
            self,
            weight_gradients: List[np.ndarray],
            bias_gradients: List[np.ndarray]
    ) -> None:
        """Update network parameters using computed gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            batch_size: Optional[int] = None,
            verbose: bool = True
    ) -> List[float]:
        """
        Train the MLP

        Args:
            X: Training data
            y: Training labels (integer vector or one-hot matrix)
            epochs: Number of training epochs
            batch_size: Size of mini-batches (None for full batch)
            verbose: Whether to print training progress

        Returns:
            List of training losses
        """
        losses = []
        m = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            batch_losses = []
            if batch_size is None:
                batches = [(X_shuffled, y_shuffled)]
            else:
                batches = [
                    (X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size])
                    for i in range(0, m, batch_size)
                ]

            for X_batch, y_batch in batches:
                activations, pre_activations = self.forward(X_batch)

                if self.output_activation == 'softmax':
                    if y_batch.ndim == 1:
                        y_onehot = np.zeros_like(activations[-1])
                        y_onehot[np.arange(y_batch.shape[0]), y_batch.astype(int)] = 1
                    else:
                        y_onehot = y_batch
                    loss = -np.sum(y_onehot * np.log(activations[-1] + 1e-15)) / y_batch.shape[0]
                else:
                    if y_batch.ndim == 2:
                        y_vec = y_batch.reshape(-1, 1)
                    else:
                        y_vec = y_batch.reshape(-1, 1)
                    loss = -np.mean(
                        y_vec * np.log(activations[-1] + 1e-15) +
                        (1 - y_vec) * np.log(1 - activations[-1] + 1e-15)
                    )
                batch_losses.append(loss)

                w_grads, b_grads = self.backward(X_batch, y_batch, activations, pre_activations)
                self.update_parameters(w_grads, b_grads)

            epoch_loss = np.mean(batch_losses)
            losses.append(epoch_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data

        Args:
            X: Input data

        Returns:
            Predicted class labels
        """
        activations, _ = self.forward(X)
        output = activations[-1]
        if self.output_activation == 'softmax':
            return np.argmax(output, axis=1)
        else:
            return (output >= 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions for input data

        Args:
            X: Input data

        Returns:
            Predicted probabilities
        """
        activations, _ = self.forward(X)
        return activations[-1]
