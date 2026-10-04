# Lilt

Detects which variety of English a text is written in.

    pip install -r requirements.txt
    python train.py     # trains, prints accuracy, writes model.joblib + metrics.json
    python app.py       # open http://127.0.0.1:5000

Varieties: American, British, Australian, Indian, Singaporean.

## Data
`data.py` holds a small hand-written starter set (24 sentences per variety).
The same 24 situations are written in every variety, so evaluation uses grouped
splits (all versions of a situation stay in the same fold). Random splits would
leak topic and score near chance.

To get meaningful numbers, replace `load_data()` with a loader for a real
corpus (ICE, GLoWbE, TwitterAAE) that returns texts, labels, groups.
