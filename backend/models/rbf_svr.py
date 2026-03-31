import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

def rbf_kernel(X, Y=None, gamma=None):
    X = np.asarray(X, dtype=float)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    distances = XX + YY - 2 * X.dot(Y.T)
    return np.exp(-gamma * distances)

def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"MSE": mse, "MAE": mae, "R2": r2}

class RBFSVRScratch(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0, epsilon=0.02, gamma=None, lr=0.001, epochs=2000, mu=100.0):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.mu = mu
        self.beta = None
        self.b = 0.0
        self.X_train = None
        self.loss_history = []

    def _train_dual(self, X, y, C, epsilon, gamma, lr, epochs, mu):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        K = rbf_kernel(X, None, gamma=gamma)
        beta = np.zeros(n, dtype=float)
        for it in range(epochs):
            Kb = K.dot(beta)
            grad = Kb + epsilon * np.sign(beta) - y + mu * np.sum(beta)
            beta = beta - lr * grad
            beta = np.clip(beta, -C, C)
            s = np.sum(beta)
            beta = beta - s / n
            beta = np.clip(beta, -C, C)
        f_train = K.dot(beta)
        b = float(np.mean(y - f_train))
        return beta, b, gamma

    def fit(self, X_train, y_train):
        self.beta, self.b, self.gamma = self._train_dual(
            X_train, y_train, C=self.C, epsilon=self.epsilon,
            gamma=self.gamma, lr=self.lr, epochs=self.epochs, mu=self.mu
        )
        self.X_train = X_train
        return self

    def predict(self, X):
        K_test = rbf_kernel(X, self.X_train, gamma=self.gamma)
        return K_test.dot(self.beta) + self.b

# 🔹 Wrapper function for pipeline
def train_rbf_svr(df, target_col):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1,1)).flatten()

    n = len(df)
    train_size = int(0.8 * n)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    model = RBFSVRScratch()
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    metrics = _metrics(y_test_unscaled, y_pred)
    predictions = {"y_test": y_test_unscaled.tolist(), "y_pred": y_pred.tolist()}

    return metrics, predictions
