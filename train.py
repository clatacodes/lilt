"""Train the dialect classifier and report accuracy.

Run:  python train.py
Writes model.joblib and metrics.json
"""
import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from data import load_data

SEED = 42


def build_pipeline():
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ]
    )
    clf = LogisticRegression(max_iter=2000, C=10.0)
    return Pipeline([("features", features), ("clf", clf)])


def main():
    texts, labels, groups = load_data()
    texts, labels, groups = np.array(texts), np.array(labels), np.array(groups)

    # 1) Held-out split (all varieties of a sentence stay on the same side)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(gss.split(texts, labels, groups))
    X_tr, X_te, y_tr, y_te = texts[tr], texts[te], labels[tr], labels[te]
    model = build_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    holdout_acc = accuracy_score(y_te, pred)
    holdout_f1 = f1_score(y_te, pred, average="macro")

    # 2) 5-fold grouped cross-validation over the whole set
    cv = GroupKFold(n_splits=5)
    cv_pred = cross_val_predict(build_pipeline(), texts, labels, cv=cv, groups=groups)
    cv_acc = accuracy_score(labels, cv_pred)
    cv_f1 = f1_score(labels, cv_pred, average="macro")

    print(f"Held-out accuracy : {holdout_acc:.3f}  (n={len(y_te)})")
    print(f"Held-out macro-F1 : {holdout_f1:.3f}")
    print(f"5-fold CV accuracy: {cv_acc:.3f}  (n={len(labels)})")
    print(f"5-fold CV macro-F1: {cv_f1:.3f}\n")
    print(classification_report(labels, cv_pred, digits=3))

    # Final model trained on everything
    final = build_pipeline().fit(texts, labels)
    joblib.dump(final, "model.joblib")

    with open("metrics.json", "w") as f:
        json.dump(
            {
                "holdout_accuracy": round(holdout_acc, 4),
                "holdout_macro_f1": round(holdout_f1, 4),
                "cv_accuracy": round(cv_acc, 4),
                "cv_macro_f1": round(cv_f1, 4),
                "n_examples": int(len(labels)),
                "n_varieties": int(len(set(labels))),
                "varieties": sorted(set(labels.tolist())),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
