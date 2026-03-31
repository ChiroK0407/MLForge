import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class LinearSVR_Scratch(BaseEstimator, RegressorMixin):
    """
    Lightweight scratch implementation of Linear SVR
    using epsilon-insensitive loss and gradient descent.
    """

    def __init__(self, lr=0.001, epochs=800, C=1.0, epsilon=0.02):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.epsilon = epsilon
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):
            y_pred = X.dot(self.w) + self.b
            error = y_pred - y

            # epsilon-insensitive loss mask
            mask = np.abs(error) > self.epsilon

            # gradients
            grad_w = self.w + (self.C / n_samples) * np.dot(X.T, error * mask)
            grad_b = self.C * np.mean(error * mask)

            # update weights
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            # loss function (hinge-like + regularization)
            loss = np.mean(np.maximum(0, np.abs(error) - self.epsilon)) + 0.5 * np.sum(self.w**2)
            self.loss_history.append(loss)

    def predict(self, X):
        return X.dot(self.w) + self.b