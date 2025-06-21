# Advanced Classification Algorithms from Scratch

A comprehensive implementation of state-of-the-art classification algorithms built entirely from first principles using NumPy. This project demonstrates mastery of machine learning theory through clean, production-ready implementations of core algorithms that power modern ML systems.

## Implemented Algorithms

### Binary & Multi-Class Classification
- **Logistic Regression** - Gradient descent with L2 regularization and numerically stable sigmoid
- **Support Vector Machine (SVM)** - Sequential Minimal Optimization (SMO) algorithm for efficient training
- **Multi-Layer Perceptron (MLP)** - Deep neural network with backpropagation and multiple activation functions
- **One-vs-Rest Classifier** - Multi-class extension using binary classifier ensembles

### Advanced Features
- **Numerical Stability**: Clipped probabilities, stable sigmoid, and robust gradient computation
- **Regularization**: L2 penalty terms to prevent overfitting
- **Optimization**: SMO for SVM, mini-batch gradient descent for neural networks
- **Multi-class Support**: Native multinomial and one-vs-rest strategies

## Technical Highlights

### Mathematical Rigor
- **Custom SMO Implementation**: Efficient quadratic optimization for SVM training
- **Backpropagation from Scratch**: Full gradient computation through neural network layers
- **Numerically Stable Operations**: Prevents overflow/underflow in exponential computations
- **Convergence Criteria**: Adaptive stopping conditions based on loss tolerance

### Production-Ready Features
- **Modular Architecture**: Clean class hierarchies with consistent interfaces
- **Memory Efficiency**: Vectorized operations and optimal data structures
- **Hyperparameter Control**: Comprehensive parameter tuning capabilities
- **Robust Error Handling**: Input validation and edge case management

## Real-World Validation

### Benchmark Datasets
- **Breast Cancer Wisconsin**: Binary classification with 30 features
- **MNIST-1D**: Multi-class digit recognition (10 classes)
- **Performance Comparison**: Direct validation against scikit-learn implementations

### Statistical Validation
- **Paired t-tests**: Rigorous statistical comparison of model performance
- **Cross-validation**: Robust performance estimation with multiple data splits
- **Custom Statistical Functions**: Implementation of t-distribution and hypothesis testing

## Advanced Implementation Details

### Support Vector Machine (SMO)
```python
# Efficient quadratic optimization without external solvers
eta = 2 * gram[i, j] - gram[i, i] - gram[j, j]
self.alpha[j] -= self.y[j] * (Ei - Ej) / eta
self.alpha[j] = np.clip(self.alpha[j], L, H)
```

### Neural Network Backpropagation
```python
# Gradient computation through network layers
delta = (activations[-1] - y_onehot) / m
weight_grads[-1] = activations[-2].T @ delta
for l in reversed(range(len(self.hidden_sizes))):
    delta = (delta @ self.weights[l + 1].T) * self._activation_derivative(pre_activations[l])
```

### Numerically Stable Sigmoid
```python
# Prevents overflow in exponential computations
pos_mask = z >= 0
result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
exp_z = np.exp(z[neg_mask])
result[neg_mask] = exp_z / (1 + exp_z)
```

## Project Architecture

```
├── logreg.py              # Logistic regression with L2 regularization
├── svm.py                 # SVM with SMO optimization algorithm
├── mlp.py                 # Multi-layer perceptron with backpropagation
├── ovr_logreg.py          # One-vs-rest multi-class wrapper
└── classification_demo.ipynb # Comprehensive testing and validation
```

## Key Learning Demonstrations

### Algorithm Mastery
- **Optimization Theory**: Implementation of constrained quadratic optimization (SMO)
- **Neural Network Theory**: Gradient descent, backpropagation, and activation functions
- **Statistical Learning**: Regularization, cross-validation, and model selection
- **Numerical Computing**: Stability, convergence, and computational efficiency

### Software Engineering Excellence
- **Clean Code**: Readable, maintainable implementations with comprehensive documentation
- **Testing & Validation**: Systematic comparison with established libraries
- **Performance Analysis**: Runtime optimization and memory management
- **API Design**: Consistent interfaces following scikit-learn conventions

## Statistical Analysis Features

### Custom T-Test Implementation
```python
def t_pdf(x: float, v: int) -> float:
    num = math.gamma((v + 1) / 2)
    denom = math.sqrt(v * math.pi) * math.gamma(v / 2)
    return num / denom * (1 + x**2 / v) ** (-(v + 1) / 2)
```

### Performance Validation
- **Accuracy Comparison**: Direct benchmarking against scikit-learn
- **Statistical Significance**: Paired t-tests for model comparison
- **Convergence Analysis**: Training loss tracking and optimization paths

## Results & Performance

### Benchmark Results
- **Breast Cancer Dataset**: >95% accuracy matching scikit-learn performance
- **MNIST-1D**: Multi-class accuracy competitive with established implementations
- **Statistical Validation**: No significant difference from reference implementations (p > 0.05)

### Computational Efficiency
- **Vectorized Operations**: Leverages NumPy's optimized linear algebra routines
- **Memory Management**: Efficient handling of large datasets and parameter matrices
- **Convergence Speed**: Optimized learning rates and stopping criteria

## Why This Matters for Employers

This project demonstrates:

- **Deep Technical Understanding**: Can implement complex ML algorithms from mathematical foundations
- **Production Code Quality**: Clean, documented, testable implementations suitable for real systems
- **Performance Optimization**: Understanding of numerical computing and efficiency considerations
- **Research Capability**: Statistical validation and rigorous experimental methodology
- **Problem-Solving Skills**: Handling numerical stability, convergence, and optimization challenges

## Getting Started

```python
# Quick demo: Multi-class classification
from ovr_logreg import OneVsRestLogisticRegression
from sklearn.metrics import accuracy_score

# Train multi-class classifier
classifier = OneVsRestLogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Evaluate performance
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.3f}")
```

---

*This project showcases the ability to bridge theoretical machine learning knowledge with practical, production-ready implementations - a critical skill for advancing ML research and applications.*
