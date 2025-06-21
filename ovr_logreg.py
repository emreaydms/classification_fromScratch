import numpy as np
from typing import Optional, Dict
from logreg import LogisticRegression


class OneVsRestLogisticRegression:
    """
    One-vs-rest strategy for multi-class classification using logistic regression.

    Each class is fitted with a separate logistic regression classifier
    to distinguish that class versus the rest.

    Parameters
    ----------
    learning_rate : float
        Learning rate for gradient descent in each binary classifier.
    max_iter : int
        Maximum number of iterations for training.
    tol : float
        Tolerance for convergence.
    C : float
        Inverse of regularization strength.
    penalty : str
        Regularization type ('l2' or 'none').
    random_state : Optional[int]
        Seed for random initialization.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        C: float = 1.0,
        penalty: str = 'l2',
        random_state: Optional[int] = None
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.penalty = penalty
        self.random_state = random_state

        self.models: Dict[int, LogisticRegression] = {}
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsRestLogisticRegression":
        """
        Fit one logistic regression classifier per class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneVsRestLogisticRegression
            Fitted classifier.
        """
        self.classes_ = np.unique(y)
        self.models.clear()

        for cls in self.classes_:
            y_binary = (y == cls).astype(int)
            clf = LogisticRegression(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                C=self.C,
                penalty=self.penalty,
                random_state=self.random_state,
            )
            clf.fit(X, y_binary)
            self.models[cls] = clf

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Probability estimates for each class.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for idx, cls in enumerate(self.classes_):
            clf = self.models[cls]
            proba[:, idx] = clf.predict_proba(X)[:, 1]

        row_sums = proba.sum(axis=1, keepdims=True)
        np.divide(proba, row_sums, out=proba, where=row_sums != 0)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        best_indices = np.argmax(proba, axis=1)
        return self.classes_[best_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy of the classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.
        y : np.ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Classification accuracy.
        """
        preds = self.predict(X)
        return np.mean(preds == y)

    def get_params(self) -> dict:
        """
        Get parameters of the underlying classifiers.

        Returns
        -------
        params : dict
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

    def set_params(self, **params) -> "OneVsRestLogisticRegression":
        """
        Set parameters of the classifiers.

        Parameters
        ----------
        params : dict
            Parameter names mapped to new values.

        Returns
        -------
        self : OneVsRestLogisticRegression
            Updated classifier.
        """
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
        return self
