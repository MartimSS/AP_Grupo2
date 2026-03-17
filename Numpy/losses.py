#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

class LossFunction:

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.shape[0]


class BinaryCrossEntropy(LossFunction):
    
    def loss(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # d/dp of mean BCE
        return (p - y_true) / (p * (1 - p) * y_true.shape[0])


class SoftmaxCrossEntropy(LossFunction):
    """Cross-entropy computed from logits (no softmax layer needed).

    Expects:
      - y_true: either shape (n,) with class indices, or one-hot shape (n, C)
      - y_pred: logits shape (n, C)
    """

    def _one_hot(self, y_true, n_classes):
        y = np.asarray(y_true).astype(int).ravel()
        oh = np.zeros((len(y), n_classes), dtype=np.float32)
        oh[np.arange(len(y)), y] = 1.0
        return oh

    def _log_softmax(self, logits):
        z = logits - np.max(logits, axis=1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(z), axis=1, keepdims=True))
        return z - logsumexp

    def loss(self, y_true, y_pred):
        logits = np.asarray(y_pred)
        n = logits.shape[0]
        log_probs = self._log_softmax(logits)

        if np.ndim(y_true) == 1:
            y_oh = self._one_hot(y_true, logits.shape[1])
        else:
            y_oh = np.asarray(y_true)

        return float(-np.sum(y_oh * log_probs) / n)

    def derivative(self, y_true, y_pred):
        logits = np.asarray(y_pred)
        n = logits.shape[0]
        log_probs = self._log_softmax(logits)
        probs = np.exp(log_probs)

        if np.ndim(y_true) == 1:
            y_oh = self._one_hot(y_true, logits.shape[1])
        else:
            y_oh = np.asarray(y_true)

        # dL/dlogits = (softmax - y) / n
        return (probs - y_oh) / n
    
    
