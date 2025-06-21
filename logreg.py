import numpy as np
from typing import Optional, Tuple, Union

class LogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        C: float = 1.0,
        penalty: str = 'l2',
        random_state: Optional[int] = None
    ):
        """
        Logistic Regression classifier using gradient descent.

        Parameters
        ----------
        learning_rate : float
            Learning rate for gradient descent.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        C : float
            Inverse of regularization strength.
        penalty : str
            Type of regularization ('l2' or 'none').
        random_state : Optional[int]
            Seed for random initialization.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.penalty = penalty
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.classes_: Optional[np.ndarray] = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically-stable sigmoid function."""
        pos_mask = z >= 0
        neg_mask = ~pos_mask
        result = np.empty_like(z, dtype=np.float64)
        result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
        exp_z = np.exp(z[neg_mask])
        result[neg_mask] = exp_z / (1 + exp_z)
        return result

    def _compute_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: float
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradient of loss w.r.t. weights and bias.

        Returns
        -------
        weight_grad : np.ndarray
            Gradient w.r.t. weights.
        bias_grad : float
            Gradient w.r.t. bias.
        """
        n_samples = X.shape[0]
        logits = X @ weights + bias
        probs = self._sigmoid(logits)
        error = probs - y
        weight_grad = (X.T @ error) / n_samples
        bias_grad = np.sum(error) / n_samples

        if self.penalty == 'l2':
            weight_grad += (1 / self.C) * weights

        return weight_grad, bias_grad

    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: float
    ) -> float:
        """
        Compute logistic loss with optional L2 regularization.

        Returns
        -------
        float
            Loss value.
        """
        n_samples = X.shape[0]
        logits = X @ weights + bias
        probs = self._sigmoid(logits)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        log_loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

        if self.penalty == 'l2':
            reg = (0.5 / self.C) * np.sum(weights ** 2)
            log_loss += reg

        return log_loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train logistic regression classifier.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        y : np.ndarray
            Training labels (0 or 1), shape (n_samples,).

        Returns
        -------
        self : LogisticRegression
            The trained model.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        prev_loss = float('inf')

        for _ in range(self.max_iter):
            loss = self._compute_loss(X, y, self.weights, self.bias)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            grad_w, grad_b = self._compute_gradient(X, y, self.weights, self.bias)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Class probabilities, shape (n_samples, 2).
        """
        logits = X @ self.weights + self.bias
        probs_pos = self._sigmoid(logits)
        probs_neg = 1 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1).
        """
        probs = self.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        if self.classes_ is not None and not np.array_equal(self.classes_, np.array([0, 1])):
            preds = self.classes_[preds]
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Parameters
        ----------
        X : np.ndarray
            Test features.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy.
        """
        preds = self.predict(X)
        return np.mean(preds == y)

    def get_params(self) -> dict:
        """
        Return model parameters.

        Returns
        -------
        dict
            Parameter dictionary.
        """
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "C": self.C,
            "penalty": self.penalty,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "LogisticRegression":
        """
        Set model parameters.

        Parameters
        ----------
        params : dict
            Parameters to set.

        Returns
        -------
        self : LogisticRegression
            Updated instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
