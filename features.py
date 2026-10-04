"""Feature pipeline: word + character n-grams plus a few handcrafted dialect cues."""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


def hand(X):
    rows = []
    for t in X:
        l = t.lower()
        w = re.findall(r"[a-z']+", l)
        n = max(len(w), 1)
        rows.append([
            len(re.findall(r"\b\w+our\b", l)),
            len(re.findall(r"\b\w+(ise|ised|ising|isation)\b", l)),
            len(re.findall(r"\b\w+(ize|ized|izing|ization)\b", l)),
            int(bool(re.search(r"\b(lah|leh|sia|lor|aiyo|makan|tapao)\b", l))),
            int(bool(re.search(r"\b(mate|reckon|heaps|arvo|servo|uni)\b", l))),
            int(bool(re.search(r"\b(only|itself|kindly|na)\b", l))),
            int(bool(re.search(r"\b(already|or not|can\?)", l))),
            len(re.findall(r"\b(am|is|are) (having|doing|going|staying|feeling)\b", l)),
            len(re.findall(r"\b(shall|whilst|brilliant|knackered|loo)\b", l)),
            sum(1 for x in w if x in ("i","you","he","she","we")) / n,
            len(w) / 20.0,
            int(t.endswith("?")),
            int("," in t),
        ])
    return np.array(rows, dtype=float)


def build_pipeline():
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                                     token_pattern=r"(?u)\b\w[\w']*\b")),
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            ("hand", FunctionTransformer(hand)),
        ]
    )
    return Pipeline([("features", features), ("clf", LogisticRegression(max_iter=2000, C=10.0))])
