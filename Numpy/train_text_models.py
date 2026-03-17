#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np
import pandas as pd

# allow running as a script
sys.path.append(os.path.dirname(__file__))

from data import Data
from text_vectorizer import CountVectorizerNumpy, TfidfVectorizerNumpy
from logistic_regression import SoftmaxRegression
from neuralnet import NeuralNetwork
from layers import DenseLayer, DropoutLayer
from activation import ReLUActivation
from losses import SoftmaxCrossEntropy
from metrics import softmax_accuracy


def load_split(csv_path: str):
    df = pd.read_csv(csv_path)
    if 'content' not in df.columns or 'model' not in df.columns:
        raise ValueError("CSV must have columns: content, model")
    texts = df['content'].astype(str).tolist()
    labels = df['model'].astype(str).tolist()
    return texts, labels


def load_examples(csv_path: str):
    """Load examples from dataset-exemplos.csv (ID;Text;Label)."""
    df = pd.read_csv(csv_path, sep=';')
    if 'Text' not in df.columns or 'Label' not in df.columns:
        raise ValueError("CSV must have columns: Text; Label")
    texts = df['Text'].astype(str).tolist()
    labels = df['Label'].astype(str).tolist()
    return texts, labels


def load_unlabeled(csv_path: str):
    """Load unlabeled texts from subm1.csv (ID;Text)."""
    df = pd.read_csv(csv_path, sep=';')
    if 'Text' not in df.columns:
        raise ValueError("CSV must have a Text column")
    ids = df['ID'].astype(str).tolist() if 'ID' in df.columns else list(range(len(df)))
    texts = df['Text'].astype(str).tolist()
    return ids, texts


def encode_labels(train_labels, labels):
    classes = sorted(set(train_labels))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = []
    keep = []
    for i, lab in enumerate(labels):
        if lab in class_to_id:
            y.append(class_to_id[lab])
            keep.append(i)
    return np.array(y, dtype=np.int64), keep, classes


def select_by_indices(items, indices):
    return [items[i] for i in indices]


def evaluate_predictions(y_true, y_pred, classes):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    acc = float(np.mean(y_true == y_pred))
    # per-class accuracy (simple diagnostic)
    per_class = {}
    for k, name in enumerate(classes):
        mask = y_true == k
        if np.any(mask):
            per_class[name] = float(np.mean(y_pred[mask] == y_true[mask]))
    return acc, per_class


def main():
    base_dir = os.path.dirname(__file__)
    ap = argparse.ArgumentParser(description="Train numpy-only text models (baseline + DNN) on your splits.")
    ap.add_argument('--train', default=os.path.normpath(os.path.join(base_dir, '..', 'splits', 'train.csv')))
    ap.add_argument('--val', default=os.path.normpath(os.path.join(base_dir, '..', 'splits', 'val.csv')))
    ap.add_argument('--test', default=os.path.normpath(os.path.join(base_dir, '..', 'splits', 'test.csv')))

    ap.add_argument('--vectorizer', choices=['count', 'tfidf'], default='tfidf')
    ap.add_argument('--max_features', type=int, default=8000)
    ap.add_argument('--min_df', type=int, default=2)

    ap.add_argument('--run', choices=['baseline', 'dnn', 'both'], default='both')

    # baseline
    ap.add_argument('--lr_base', type=float, default=0.2)
    ap.add_argument('--epochs_base', type=int, default=150)
    ap.add_argument('--batch_base', type=int, default=512)
    ap.add_argument('--l2_base', type=float, default=1e-4)

    # dnn
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr_dnn', type=float, default=0.05)
    ap.add_argument('--epochs_dnn', type=int, default=60)
    ap.add_argument('--batch_dnn', type=int, default=256)
    ap.add_argument('--l2_dnn', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=8)

    args = ap.parse_args()

    train_texts, train_labels = load_split(args.train)
    val_texts, val_labels = load_split(args.val)
    test_texts, test_labels = load_split(args.test)

    y_train, keep_train, classes = encode_labels(train_labels, train_labels)
    y_val, keep_val, _ = encode_labels(train_labels, val_labels)
    y_test, keep_test, _ = encode_labels(train_labels, test_labels)

    train_texts = select_by_indices(train_texts, keep_train)
    val_texts = select_by_indices(val_texts, keep_val)
    test_texts = select_by_indices(test_texts, keep_test)

    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")

    if args.vectorizer == 'count':
        vec = CountVectorizerNumpy(max_features=args.max_features, min_df=args.min_df)
    else:
        vec = TfidfVectorizerNumpy(max_features=args.max_features, min_df=args.min_df)

    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)
    X_test = vec.transform(test_texts)

    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    if args.run in ('baseline', 'both'):
        print("\n== Baseline: Softmax Regression ==")
        base = SoftmaxRegression(
            learning_rate=args.lr_base,
            epochs=args.epochs_base,
            batch_size=args.batch_base,
            l2=args.l2_base,
            verbose=True,
        )
        base.fit(X_train, y_train, n_classes=n_classes, X_val=X_val, y_val=y_val, patience=10, min_delta=1e-4)

        pred_val = base.predict(X_val)
        pred_test = base.predict(X_test)
        acc_val, _ = evaluate_predictions(y_val, pred_val, classes)
        acc_test, per_class = evaluate_predictions(y_test, pred_test, classes)
        print(f"Val accuracy:  {acc_val:.4f}")
        print(f"Test accuracy: {acc_test:.4f}")
        print("Per-class test accuracy:")
        for k, v in per_class.items():
            print(f"  {k}: {v:.4f}")

        # Evaluate on dataset-exemplos.csv
        examples_path = os.path.normpath(os.path.join(base_dir, '..', 'dataset-exemplos.csv'))
        if os.path.exists(examples_path):
            ex_texts, ex_labels = load_examples(examples_path)
            y_ex, keep_ex, _ = encode_labels(train_labels, ex_labels)
            ex_texts = select_by_indices(ex_texts, keep_ex)
            if len(ex_texts) > 0:
                X_ex = vec.transform(ex_texts)
                pred_ex = base.predict(X_ex)
                acc_ex, per_class_ex = evaluate_predictions(y_ex, pred_ex, classes)
                print("\n== Baseline on dataset-exemplos.csv ==")
                print(f"Accuracy: {acc_ex:.4f}")
                print("Per-class accuracy:")
                for k, v in per_class_ex.items():
                    print(f"  {k}: {v:.4f}")

        # Predict for subm1.csv and save predictions
        subm_path = os.path.normpath(os.path.join(base_dir, '..', 'subm1.csv'))
        if os.path.exists(subm_path):
            ids_sub, texts_sub = load_unlabeled(subm_path)
            if len(texts_sub) > 0:
                X_sub = vec.transform(texts_sub)
                pred_sub_idx = base.predict(X_sub)
                pred_sub_labels = [classes[i] for i in pred_sub_idx]
                out_df = pd.DataFrame({
                    'ID': ids_sub,
                    'Text': texts_sub,
                    'PredictedModel': pred_sub_labels,
                })
                out_path = os.path.normpath(os.path.join(base_dir, '..', 'subm1_predictions_baseline.csv'))
                out_df.to_csv(out_path, index=False)
                print(f"Baseline predictions written to {out_path}")

    if args.run in ('dnn', 'both'):
        print("\n== DNN (numpy) ==")
        ds_train = Data(X_train.astype(np.float32), y_train)
        ds_val = Data(X_val.astype(np.float32), y_val)
        ds_test = Data(X_test.astype(np.float32), y_test)

        net = NeuralNetwork(
            epochs=args.epochs_dnn,
            batch_size=args.batch_dnn,
            learning_rate=args.lr_dnn,
            verbose=True,
            loss=SoftmaxCrossEntropy,
            metric=softmax_accuracy,
        )

        n_features = ds_train.X.shape[1]
        net.add(DenseLayer(args.hidden, (n_features,), l2_lambda=args.l2_dnn))
        net.add(ReLUActivation())
        net.add(DropoutLayer(p_drop=args.dropout))
        net.add(DenseLayer(n_classes, l2_lambda=args.l2_dnn))  # logits

        net.fit(ds_train, val_dataset=ds_val, patience=args.patience, min_delta=1e-4)

        logits_test = net.predict(ds_test)
        pred_test = np.argmax(logits_test, axis=1)
        acc_test, per_class = evaluate_predictions(y_test, pred_test, classes)
        print(f"Test accuracy: {acc_test:.4f}")
        print("Per-class test accuracy:")
        for k, v in per_class.items():
            print(f"  {k}: {v:.4f}")

        # Evaluate DNN on dataset-exemplos.csv
        examples_path = os.path.normpath(os.path.join(base_dir, '..', 'dataset-exemplos.csv'))
        if os.path.exists(examples_path):
            ex_texts, ex_labels = load_examples(examples_path)
            y_ex, keep_ex, _ = encode_labels(train_labels, ex_labels)
            ex_texts = select_by_indices(ex_texts, keep_ex)
            if len(ex_texts) > 0:
                X_ex = vec.transform(ex_texts).astype(np.float32)
                ds_ex = Data(X_ex, y_ex)
                logits_ex = net.predict(ds_ex)
                pred_ex = np.argmax(logits_ex, axis=1)
                acc_ex, per_class_ex = evaluate_predictions(y_ex, pred_ex, classes)
                print("\n== DNN on dataset-exemplos.csv ==")
                print(f"Accuracy: {acc_ex:.4f}")
                print("Per-class accuracy:")
                for k, v in per_class_ex.items():
                    print(f"  {k}: {v:.4f}")

        # Predict for subm1.csv with DNN and save predictions
        subm_path = os.path.normpath(os.path.join(base_dir, '..', 'subm1.csv'))
        if os.path.exists(subm_path):
            ids_sub, texts_sub = load_unlabeled(subm_path)
            if len(texts_sub) > 0:
                X_sub = vec.transform(texts_sub).astype(np.float32)
                ds_sub = Data(X_sub)
                logits_sub = net.predict(ds_sub)
                pred_sub_idx = np.argmax(logits_sub, axis=1)
                pred_sub_labels = [classes[i] for i in pred_sub_idx]
                out_df = pd.DataFrame({
                    'ID': ids_sub,
                    'Text': texts_sub,
                    'PredictedModel': pred_sub_labels,
                })
                out_path = os.path.normpath(os.path.join(base_dir, '..', 'subm1_predictions_dnn.csv'))
                out_df.to_csv(out_path, index=False)
                print(f"DNN predictions written to {out_path}")


if __name__ == '__main__':
    main()
