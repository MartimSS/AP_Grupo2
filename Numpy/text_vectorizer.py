#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np


_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def simple_tokenize(text: str, lowercase: bool = True):
    if text is None:
        return []
    text = str(text)
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


class CountVectorizerNumpy:
    """Minimal CountVectorizer implemented with numpy.

    Produces dense matrices; keep max_features reasonably small (e.g. 2k-20k).
    """

    def __init__(self, max_features: int = 5000, min_df: int = 1, lowercase: bool = True, dtype=np.float32):
        self.max_features = int(max_features) if max_features is not None else None
        self.min_df = int(min_df)
        self.lowercase = lowercase
        self.dtype = dtype
        self.vocab_ = None
        self.idf_ = None

    def fit(self, texts):
        doc_freq = {}
        for text in texts:
            tokens = set(simple_tokenize(text, lowercase=self.lowercase))
            for tok in tokens:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        # apply min_df
        items = [(tok, df) for tok, df in doc_freq.items() if df >= self.min_df]
        # sort by df desc, then token
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.max_features is not None:
            items = items[: self.max_features]

        self.vocab_ = {tok: i for i, (tok, _) in enumerate(items)}
        return self

    def transform(self, texts):
        if self.vocab_ is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        n_samples = len(texts)
        n_features = len(self.vocab_)
        X = np.zeros((n_samples, n_features), dtype=self.dtype)

        for i, text in enumerate(texts):
            for tok in simple_tokenize(text, lowercase=self.lowercase):
                j = self.vocab_.get(tok)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


class TfidfVectorizerNumpy(CountVectorizerNumpy):
    """TF-IDF Vectorizer using a CountVectorizer base, implemented with numpy."""

    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 1,
        lowercase: bool = True,
        dtype=np.float32,
        norm = "l2",
        smooth_idf: bool = True,
    ):
        super().__init__(max_features=max_features, min_df=min_df, lowercase=lowercase, dtype=dtype)
        self.norm = norm
        self.smooth_idf = smooth_idf

    def fit(self, texts):
        super().fit(texts)
        n_docs = len(texts)
        df = np.zeros(len(self.vocab_), dtype=np.int32)

        for text in texts:
            seen = set()
            for tok in simple_tokenize(text, lowercase=self.lowercase):
                j = self.vocab_.get(tok)
                if j is None or j in seen:
                    continue
                df[j] += 1
                seen.add(j)

        if self.smooth_idf:
            self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        else:
            # Avoid div by 0
            df = np.maximum(df, 1)
            self.idf_ = np.log(n_docs / df) + 1.0
        self.idf_ = self.idf_.astype(self.dtype)
        return self

    def transform(self, texts):
        X = super().transform(texts)
        if self.idf_ is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        # TF: normalize by doc length (sum of counts)
        doc_len = np.sum(X, axis=1, keepdims=True)
        doc_len = np.maximum(doc_len, 1.0)
        tf = X / doc_len

        X_tfidf = tf * self.idf_[None, :]

        if self.norm == "l2":
            norms = np.linalg.norm(X_tfidf, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X_tfidf = X_tfidf / norms
        elif self.norm is None:
            pass
        else:
            raise ValueError("norm must be 'l2' or None")

        return X_tfidf.astype(self.dtype)
