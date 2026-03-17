#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from optimizer import Optimizer


def _softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _one_hot(y, n_classes):
    y = np.asarray(y).astype(int).ravel()
    oh = np.zeros((len(y), n_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


class SoftmaxRegression:
    """Multiclass logistic regression (softmax regression) trained with minibatch GD."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        epochs: int = 200,
        batch_size: int = 256,
        l2: float = 0.0,
        verbose: bool = False,
        seed: int = 42,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.verbose = verbose
        self.seed = seed

        self.W = None
        self.b = None
        self.w_opt = Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.b_opt = Optimizer(learning_rate=learning_rate, momentum=momentum)

        self.history = {}

    def _iter_batches(self, X, y):
        n = X.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(idx)
        for start in range(0, n, self.batch_size):
            sel = idx[start : start + self.batch_size]
            yield X[sel], y[sel]

    def fit(self, X, y, n_classes=None, X_val=None, y_val=None, patience: int = 0, min_delta: float = 0.0):
        X = np.asarray(X)
        y = np.asarray(y).ravel().astype(int)
        n_samples, n_features = X.shape
        if n_classes is None:
            n_classes = int(np.max(y)) + 1

        rng = np.random.default_rng(self.seed)
        self.W = (rng.standard_normal((n_features, n_classes)).astype(np.float32)) * 0.01
        self.b = np.zeros((1, n_classes), dtype=np.float32)

        best = None
        best_val = np.inf
        bad_epochs = 0

        for epoch in range(1, self.epochs + 1):
            losses = []
            for Xb, yb in self._iter_batches(X, y):
                y_oh = _one_hot(yb, n_classes)
                logits = Xb @ self.W + self.b
                probs = _softmax(logits)

                # cross-entropy mean
                p = np.clip(probs, 1e-15, 1 - 1e-15)
                loss = -np.mean(np.sum(y_oh * np.log(p), axis=1))
                if self.l2 > 0:
                    loss += 0.5 * self.l2 * np.sum(self.W * self.W)
                losses.append(loss)

                grad_logits = (probs - y_oh) / Xb.shape[0]
                grad_W = Xb.T @ grad_logits
                grad_b = np.sum(grad_logits, axis=0, keepdims=True)
                if self.l2 > 0:
                    grad_W += self.l2 * self.W

                self.W = self.w_opt.update(self.W, grad_W)
                self.b = self.b_opt.update(self.b, grad_b)

            train_loss = float(np.mean(losses)) if losses else float('nan')
            self.history[epoch] = {"loss": train_loss}

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._loss(X_val, y_val, n_classes)
                self.history[epoch]["val_loss"] = val_loss

                if val_loss < (best_val - min_delta):
                    best_val = val_loss
                    best = (self.W.copy(), self.b.copy())
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if patience and bad_epochs >= patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if self.verbose and (epoch == 1 or epoch % 10 == 0):
                if val_loss is None:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f}")
                else:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if best is not None:
            self.W, self.b = best
        return self

    def _loss(self, X, y, n_classes):
        y = np.asarray(y).ravel().astype(int)
        y_oh = _one_hot(y, n_classes)
        probs = self.predict_proba(X)
        p = np.clip(probs, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_oh * np.log(p), axis=1))
        if self.l2 > 0:
            loss += 0.5 * self.l2 * np.sum(self.W * self.W)
        return float(loss)

    def predict_proba(self, X):
        X = np.asarray(X)
        logits = X @ self.W + self.b
        return _softmax(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
