import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin

class PolySVR_Scratch(BaseEstimator, RegressorMixin):
    """
    Polynomial SVR (linear epsilon-SVR in primal form) with proper preprocessing:
    - StandardScaler on base features
    - PolynomialFeatures expansion
    - StandardScaler on expanded features
    - StandardScaler on target y
    """

    def __init__(self, lr=0.001, epochs=900, C=1.0, epsilon=0.02, degree=2, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.epsilon = epsilon
        self.degree = degree
        self.verbose = verbose

        # Learned parameters
        self.w = None
        self.b = 0.0
        self.loss_history = []

        # Preprocessors
        self.base_scaler = StandardScaler()
        self.poly_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)

    def _epsilon_loss(self, error):
        return np.maximum(0, np.abs(error) - self.epsilon)

    def fit(self, X, y):
        # Scale base features
        X_base = self.base_scaler.fit_transform(X)
        # Polynomial expansion
        X_poly = self.poly.fit_transform(X_base)
        # Scale expanded features
        X_poly = self.poly_scaler.fit_transform(X_poly)
        # Scale target
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        n_samples, n_features = X_poly.shape
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):
            y_pred = X_poly.dot(self.w) + self.b
            error = y_pred - y_scaled

            mask = (np.abs(error) > self.epsilon).astype(float)
            direction = np.sign(error)

            grad_w = self.w + self.C * (mask * direction) @ X_poly / n_samples
            grad_b = self.C * np.mean(mask * direction)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            loss = 0.5 * np.dot(self.w, self.w) + self.C * np.mean(self._epsilon_loss(error))
            self.loss_history.append(loss)

            if self.verbose and epoch % 50 == 0:
                print(f"Epoch {epoch} | Loss = {loss:.6f}")

        if self.verbose:
            print("Training finished.")

    def predict(self, X):
        # Apply same preprocessing
        X_base = self.base_scaler.transform(X)
        X_poly = self.poly.transform(X_base)
        X_poly = self.poly_scaler.transform(X_poly)

        y_pred_scaled = X_poly.dot(self.w) + self.b
        return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 🔹 Wrapper function for pipeline integration
def train_poly_svr(df, target_col, degree=2):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    n = len(df)
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = PolySVR_Scratch(lr=0.001, epochs=900, C=1.0, epsilon=0.02, degree=degree)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = float(np.mean((y_pred - y_test) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_test)))
    r2 = float(1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    metrics = {"MSE": mse, "MAE": mae, "R2": r2}
    predictions = {
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }

    return metrics, predictions
