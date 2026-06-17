# HeartCare AI — Flask Frontend

This project provides a Flask frontend for a heart disease prediction model.

Quick setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Prepare the trained model:

- If you already have `best_model.pkl`, place it in the project root (`d:\Antigravity\projects\medicare\best_model.pkl`).
- Or run the included training script (ensure `heart.csv` is in the same folder or update the path):

```bash
python heart_disease.py
```

This will create `best_model.pkl` using the best-performing model.

3. Run the Flask app:

```bash
python app.py
```

4. Open `http://127.0.0.1:5000` in your browser.

Notes

- The Flask app will show a "Model unavailable" message if `best_model.pkl` is not present.
- You can replace the heuristic logic in `app.py` with the model prediction (already wired to load `best_model.pkl` if present).

If you want, I can also add a web UI endpoint to upload a `best_model.pkl` directly from the browser.
