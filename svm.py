import numpy as np
from typing import Optional, Tuple

class SVM:
    def __init__(
        self,
        C: float = 1.0,
        tol: float = 1e-3,
        max_passes: int = 5,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ):
        """
        Simplified SVM classifier using SMO for linear kernel.

        Parameters
        ----------
        C : float
            Regularization parameter.
        tol : float
            Numerical tolerance.
        max_passes : int
            Maximum number of passes over alpha pairs without changes before stopping.
        max_iter : int
            Maximum number of total iterations.
        random_state : int, optional
            Seed for reproducibility.
        """
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state

        self.alpha: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.w: Optional[np.ndarray] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Fit the SVM model on training data.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,), should be -1 or 1.

        Returns
        -------
        self : SVM
            Fitted classifier.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        m, n = X.shape
        self.X = X
        self.y = y.astype(float)
        self.alpha = np.zeros(m)
        self.b = 0.0

        gram = X @ X.T
        passes = 0
        iters = 0

        while passes < self.max_passes and iters < self.max_iter:
            num_changed = 0
            for i in range(m):
                Ei = self._decision(X[i]) - self.y[i]

                if (self.y[i] * Ei < -self.tol and self.alpha[i] < self.C) or (
                        self.y[i] * Ei > self.tol and self.alpha[i] > 0
                ):
                    j = self._pick_other_index(i, m)
                    Ej = self._decision(X[j]) - self.y[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    if self.y[i] == self.y[j]:
                        L = max(0, alpha_j_old + alpha_i_old - self.C)
                        H = min(self.C, alpha_j_old + alpha_i_old)
                    else:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)

                    if L == H:
                        continue

                    eta = 2 * gram[i, j] - gram[i, i] - gram[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= self.y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        self.alpha[j] = alpha_j_old
                        continue

                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    b1 = (
                            self.b
                            - Ei
                            - self.y[i] * (self.alpha[i] - alpha_i_old) * gram[i, i]
                            - self.y[j] * (self.alpha[j] - alpha_j_old) * gram[i, j]
                    )
                    b2 = (
                            self.b
                            - Ej
                            - self.y[i] * (self.alpha[i] - alpha_i_old) * gram[i, j]
                            - self.y[j] * (self.alpha[j] - alpha_j_old) * gram[j, j]
                    )

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

                    num_changed += 1

            iters += 1
            passes = passes + 1 if num_changed == 0 else 0

        self.w = ((self.alpha * self.y).reshape(-1, 1) * X).sum(axis=0)
        return self

    def _pick_other_index(self, i: int, n: int) -> int:
        """
        Pick a random index j ≠ i.

        Parameters
        ----------
        i : int
            First index.
        n : int
            Total number of samples.

        Returns
        -------
        j : int
            Second index.
        """
        j = i
        while j == i:
            j = np.random.randint(0, n)
        return j

    def _decision(self, x: np.ndarray) -> float:
        """
        Internal helper to compute decision function.

        Parameters
        ----------
        x : np.ndarray
            Input sample.

        Returns
        -------
        float
            Raw decision function value.
        """
        if self.w is not None:
            return float(x @ self.w + self.b)
        return float((self.alpha * self.y) @ (self.X @ x) + self.b)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Decision function values.
        """
        if self.w is not None:
            return X @ self.w + self.b
        return (self.alpha * self.y) @ (self.X @ X.T) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted labels (-1 or 1).
        """
        return np.sign(self.decision_function(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy of predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy score.
        """
        preds = self.predict(X)
        return np.mean(preds == y)
