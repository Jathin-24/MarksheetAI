"""Train the always-available digit classifier (scikit-learn MLP) on MNIST with handwriting augmentation."""
from __future__ import annotations

import io
import logging
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from config import settings

log = logging.getLogger(__name__)
MNIST_URL = "https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"
MODEL_PATH = settings.MODEL_DIR / "mnist_mlp.joblib"


def load_mnist(per_class: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Download MNIST (png archive on GitHub, ~15 MB) once and cache the subset next to the models."""
    cache = settings.MODEL_DIR / "mnist_subset.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]
    log.info("downloading MNIST ...")
    data = urllib.request.urlopen(MNIST_URL, timeout=180).read()
    counts = {d: 0 for d in range(10)}
    xs, ys = [], []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf:
            parts = Path(m.name).parts
            if not (m.isfile() and "training" in parts and m.name.endswith(".png")):
                continue
            d = int(parts[-2])
            if counts[d] >= per_class:
                continue
            img = cv2.imdecode(np.frombuffer(tf.extractfile(m).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            xs.append(img); ys.append(d); counts[d] += 1
    X, y = np.array(xs, np.uint8), np.array(ys)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random rotation/shear/scale + thinner or bolder strokes, to mimic pen handwriting photographed in red."""
    ang = rng.uniform(-14, 14)
    sh = rng.uniform(-0.25, 0.25)
    sc = rng.uniform(0.85, 1.1)
    m = cv2.getRotationMatrix2D((14, 14), ang, sc)
    m[0, 1] += sh
    out = cv2.warpAffine(img, m, (28, 28), flags=cv2.INTER_LINEAR)
    r = rng.random()
    if r < 0.3:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    elif r < 0.55:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))
    return out


def train(out_path: Path = MODEL_PATH, copies: int = 3, extra: tuple[np.ndarray, np.ndarray] | None = None):
    import joblib
    from sklearn.neural_network import MLPClassifier
    X, y = load_mnist()
    if extra is not None:                       # teacher-verified crops (see scripts/retrain_from_labels.py)
        X, y = np.concatenate([X, np.repeat(extra[0], 20, 0)]), np.concatenate([y, np.repeat(extra[1], 20)])
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for img, lab in zip(X, y):
        xs.append(img); ys.append(lab)
        for _ in range(copies):
            xs.append(augment(img, rng)); ys.append(lab)
    A = np.array(xs, np.float32).reshape(len(xs), -1) / 255.0
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, max_iter=25, early_stopping=True,
                        random_state=0, verbose=False)
    clf.fit(A, np.array(ys))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path, compress=3)
    return clf
