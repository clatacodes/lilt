"""Run:  python app.py   then open http://127.0.0.1:5000"""
import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(HERE, "model.joblib"))
with open(os.path.join(HERE, "metrics.json")) as f:
    metrics = json.load(f)

features = model.named_steps["features"]
clf = model.named_steps["clf"]
word_vec = dict(features.transformer_list)["word"]
word_names = word_vec.get_feature_names_out()
n_word = len(word_names)

app = Flask(__name__)


def top_clues(text, class_idx, k=6):
    """Words/phrases in the text that push the score toward the predicted variety."""
    x = word_vec.transform([text])
    coefs = clf.coef_[class_idx][:n_word]
    contrib = x.multiply(coefs).tocoo()
    items = sorted(zip(contrib.col, contrib.data), key=lambda t: -t[1])
    return [str(word_names[i]) for i, v in items if v > 0][:k]


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if len(text.split()) < 3:
        return jsonify(error="Enter at least three words."), 400
    probs = model.predict_proba([text])[0]
    order = np.argsort(-probs)
    classes = clf.classes_
    return jsonify(
        results=[{"variety": str(classes[i]), "probability": float(probs[i])} for i in order],
        clues=top_clues(text, int(order[0])),
    )


if __name__ == "__main__":
    app.run(debug=False)
