"""Portfolio adaptation of ENGI 9808 Lab 1: utensil image classification.

Requires locally supplied images in data/FORK, data/KNIFE and data/SPOON.
This demonstrator uses 5-fold CV on development data, not the full lab sweep.
"""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

CLASSES = ("FORK", "KNIFE", "SPOON")


def load_features(root: Path):
    features, labels = [], []
    for label, name in enumerate(CLASSES):
        paths = sorted((root / name).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNG images in {root / name}")
        for path in paths:
            image = imread(path)
            if image.ndim == 3 and image.shape[2] == 4:
                image = rgba2rgb(image)
            if image.ndim == 3:
                image = rgb2gray(image)
            image = resize(image, (64, 64), anti_aliasing=True)
            features.append(hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
            labels.append(label)
    return np.asarray(features), np.asarray(labels)


def main(root: Path):
    X, y = load_features(root)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    candidates = {
        "linear": SVC(kernel="linear", C=1.0),
        "polynomial": SVC(kernel="poly", C=1.0, degree=2, gamma=0.1),
        "rbf": SVC(kernel="rbf", C=3.0, gamma=0.1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {
        name: float(cross_val_score(model, X_dev, y_dev, cv=cv, scoring="accuracy").mean())
        for name, model in candidates.items()
    }
    for name, score in scores.items():
        print(f"{name}: mean 5-fold validation accuracy = {score:.2%}")
    selected = max(scores, key=scores.get)
    model = candidates[selected].fit(X_dev, y_dev)
    predicted = model.predict(X_test)
    print(f"Selected kernel: {selected}")
    print(f"Held-out test accuracy: {accuracy_score(y_test, predicted):.2%}")
    print("Confusion matrix:\n", confusion_matrix(y_test, predicted, labels=[0, 1, 2]))
    print(classification_report(y_test, predicted, labels=[0, 1, 2], target_names=CLASSES, zero_division=0))
    print("Note: This redesigned evaluation is not the original lab's reported experiment.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    main(args.data)
